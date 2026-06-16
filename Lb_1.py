import cv2
import numpy as np
import math
import sys
import os

# 1. ИМПОРТ МОДУЛЬНЫХ ФУНКЦИЙ (добавлена calculate_tracking_metrics)
from vision_func import (
    load_calibration, setup_aruco_detector, detect_obstacles, detect_robot,
    calculate_navigation, draw_overlay, MIN_OBSTACLE_AREA, COLOR_RANGES, draw_reachable_zone
)
from robot_func import connect_to_robotino, send_velocity, stop_robot, calculate_robot_velocities
from path_planner import plan_path, generate_grid_for_debug, calculate_tracking_metrics
from config_loader import load_experiment_config

# ================= ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ =================
target_px = None
target_m = None
current_path = []
planned_path_snapshot = []  # Снимок запланированного пути для сравнения
actual_path = []            # Фактическая траектория движения робота
metrics_printed = False     # Флаг, чтобы не спамить метриками каждый кадр после финиша
sock = None

def create_mouse_callback(calib):
    def callback(event, x, y, flags, param):
        global target_px, target_m, current_path, planned_path_snapshot, actual_path, metrics_printed
        if event == cv2.EVENT_LBUTTONDOWN:
            if cv2.pointPolygonTest(calib["field_poly"], (x, y), False) < 0:
                print("[WARNING] Клик за пределами поля.")
                return
            target_px = (x, y)
            pt_top = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), calib["M"])
            target_m = (pt_top[0][0][0] / calib["px_x"], pt_top[0][0][1] / calib["px_y"])
            
            # Сброс всех путей и флага при новой цели
            current_path = []
            planned_path_snapshot = []
            actual_path = []
            metrics_printed = False
            print(f"[INFO] Цель задана: ({target_m[0]:.2f}, {target_m[1]:.2f} м). Ожидание пути...")
    return callback

def main():
    global target_m, current_path, planned_path_snapshot, actual_path, metrics_printed, sock
    
    print("[INFO] Инициализация системы (Лабораторная 1 - Базовое движение с метриками)...")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.yaml")
    
    # 2. Загрузка конфигурации
    try:
        cfg = load_experiment_config(config_path)
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки конфигурации: {e}")
        sys.exit(1)

    robot = cfg["robot"]
    nav = cfg["navigation"]
    pp = cfg["path_planner"]

    # 3. Формирование единого словаря для планировщика
    planner_config = {
        "type": pp["type"],
        "cell_size": pp["cell_size"],
        "safety_margin": pp["safety_margin"],
        "obs_radius_m": pp["obs_radius_m"],
        "rrt_star": pp["rrt_star"],
        "apf": pp["apf"]
    }

    print("=" * 50)
    print(f"АЛГОРИТМ: {planner_config['type'].upper()}")
    print(f"РАЗМЕР ЯЧЕЙКИ: {pp['cell_size']} м")
    print("=" * 50)

    # 4. Инициализация оборудования
    calib = load_calibration(config_path, 1000, 800)
    detector = setup_aruco_detector()
    sock = connect_to_robotino(robot["ip"], robot["port"])
    
    if sock is None:
        print("[WARNING] Не удалось подключиться к роботу. Работа в режиме визуальной отладки.")
    else:
        print("[INFO] TCP подключение к роботу установлено.")

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] Не удалось открыть камеру.")
        sys.exit(1)

    window_name = "Field & Objects"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 700)
    cv2.setMouseCallback(window_name, create_mouse_callback(calib))

    print("[INFO] Готово. Кликните ЛКМ для выбора цели. 'q' или ESC для выхода.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Потеря кадра.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 5. Детектирование
        robot_data = detect_robot(
            gray, detector, calib["M"], calib["M_inv"], 
            calib["px_x"], calib["px_y"], robot["diameter_m"], calib["field_poly"]
        )
        aruco_info = {"center": robot_data["center_px"], "radius": robot_data["radius_px"]} if robot_data else None

        obstacles = detect_obstacles(
            hsv, calib["M"], calib["px_x"], calib["px_y"], calib["field_poly"], 
            MIN_OBSTACLE_AREA, COLOR_RANGES,
            aruco_center=aruco_info["center"] if aruco_info else None,
            aruco_radius=aruco_info["radius"] if aruco_info else 0
        )

        # 6. Построение пути (только один раз при появлении новой цели)
        if not current_path and target_m and robot_data:
            dist_to_target = math.hypot(target_m[0] - robot_data["center_m"][0], 
                                        target_m[1] - robot_data["center_m"][1])
            
            if dist_to_target > pp["waypoint_threshold"] * 1.5:
                new_path = plan_path(
                    robot_data["center_m"], target_m,
                    calib["data"]["width_m"], calib["data"]["height_m"],
                    obstacles, planner_config
                )
                
                if new_path is not None:
                    current_path = new_path
                    planned_path_snapshot = list(new_path) # Сохраняем снимок плана для будущих метрик
                    print(f"[INFO] Путь построен: {len(current_path)} точек")
                else:
                    print("[ERROR] Путь не построен! Робот остановлен.")
            else:
                current_path = []

        # 7. Следование по пути и сбор фактической траектории
        Vx_robot, Vy_robot = 0.0, 0.0
        if current_path and robot_data:
            current_pos = robot_data["center_m"]
            
            # Записываем фактическую позицию (с порогом 1 см, чтобы не дублировать точки)
            if not actual_path or math.hypot(current_pos[0] - actual_path[-1][0], current_pos[1] - actual_path[-1][1]) > 0.01:
                actual_path.append(current_pos)

            next_wp = current_path[0]
            dist_to_wp = math.hypot(next_wp[0] - current_pos[0], next_wp[1] - current_pos[1])

            if dist_to_wp < pp["waypoint_threshold"]:
                current_path.pop(0)
                
                # ================= МАРШРУТ ЗАВЕРШЕН =================
                if not current_path and not metrics_printed:
                    print("[INFO] Цель достигнута!")
                    
                    # Расчет и вывод метрик
                    if actual_path and planned_path_snapshot:
                        actual_mm, planned_mm, mse, r2 = calculate_tracking_metrics(actual_path, planned_path_snapshot)
                        if actual_mm is not None:
                            print("-" * 50)
                            print("[МЕТРИКИ КАЧЕСТВА СЛЕДОВАНИЯ ТРАЕКТОРИИ]")
                            print(f"  Планируемая длина: {planned_mm:7.1f} мм")
                            print(f"  Фактический путь:  {actual_mm:7.1f} мм")
                            print(f"  MSE (отклонение):  {mse:7.2f} мм²")
                            print(f"  R² (точность):     {r2:7.4f}")
                            print("-" * 50)
                    
                    metrics_printed = True
            else:
                # Кинематика: преобразование скоростей из глобальной в локальную систему робота
                Vx_robot, Vy_robot = calculate_robot_velocities(robot_data, next_wp, nav)

        # 8. Отправка команд
        if sock:
            send_velocity(Vx_robot, Vy_robot, 0.0, robot["ip"])

        # 9. Отрисовка
        grid_current = generate_grid_for_debug(
            calib["data"]["width_m"], calib["data"]["height_m"], 
            obstacles, pp["cell_size"], pp["safety_margin"], pp["obs_radius_m"]
        )
        draw_reachable_zone(
            frame, grid_current, 
            robot_data["center_m"] if robot_data else None, 
            calib, pp["cell_size"], alpha=0.25
        )

        nav_data = calculate_navigation(robot_data, target_m) if (robot_data and target_m) else None 
        
        draw_overlay(
            frame, calib, obstacles, robot_data, target_px, nav_data, target_m, 
            current_path=current_path, 
            safety_margin_m=pp["safety_margin"]
        )

        cv2.imshow(window_name, frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    # 10. Завершение работы
    cap.release()
    cv2.destroyAllWindows()
    if sock:
        stop_robot(robot["ip"])
    print("[INFO] Программа завершена.")

if __name__ == "__main__":
    main()