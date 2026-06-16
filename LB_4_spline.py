import cv2
import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 1. ИМПОРТ МОДУЛЬНЫХ ФУНКЦИЙ
from config_loader import load_experiment_config
from vision_func import (
    load_calibration, setup_aruco_detector, detect_obstacles, detect_robot,
    calculate_navigation, draw_overlay, MIN_OBSTACLE_AREA, COLOR_RANGES, draw_reachable_zone
)
from robot_func import connect_to_robotino, send_velocity, stop_robot, calculate_robot_velocities
from path_planner import plan_path, generate_grid_for_debug, smooth_path_spline, calculate_tracking_metrics

# ================= ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ =================
target_px = None
target_m = None
current_path = []          # Сглаженный путь (по нему едет робот, очищается по мере движения)
original_path = []         # Исходный "сырой" путь (от планировщика)
smoothed_path_snapshot = [] # Снимок сглаженного пути (сохраняется для графиков и метрик)
actual_path = []           # Фактическая траектория робота
sock = None
metrics_printed = False    # Флаг, чтобы не печатать метрики и графики каждый кадр после финиша

def create_mouse_callback(calib):
    def callback(event, x, y, flags, param):
        global target_px, target_m, current_path, original_path, smoothed_path_snapshot, actual_path, metrics_printed
        if event == cv2.EVENT_LBUTTONDOWN:
            if cv2.pointPolygonTest(calib["field_poly"], (x, y), False) < 0:
                print("[WARNING] Клик за пределами поля.")
                return
            target_px = (x, y)
            pt_top = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), calib["M"])
            target_m = (pt_top[0][0][0] / calib["px_x"], pt_top[0][0][1] / calib["px_y"])
            
            # Сброс всех путей и флага при новой цели
            current_path = []
            original_path = []
            smoothed_path_snapshot = []
            actual_path = []
            metrics_printed = False
            print(f"[INFO] Цель задана: ({target_m[0]:.2f}, {target_m[1]:.2f} м). Ожидание пути...")
    return callback

def save_trajectory_plots(actual_path, smoothed_path, original_path, algo_name):
    """
    Строит и сохраняет графики зависимости X и Y от индекса точки пути.
    """
    os.makedirs("screenshots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/trajectory_plot_{algo_name}_{timestamp}.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'Траектория движения (Алгоритм: {algo_name.upper()})', fontsize=16)

    # График 1: X от индекса
    ax1.plot([p[0] for p in original_path], label='Исходный путь (Raw)', linestyle='--', color='gray', marker='.', markersize=4)
    ax1.plot([p[0] for p in smoothed_path], label='Сглаженный путь (Spline)', linestyle='-', color='orange', linewidth=2, marker='o', markersize=3)
    #ax1.plot([p[0] for p in actual_path], label='Фактическая траектория (Robot)', linestyle='-', color='blue', alpha=0.7)
    ax1.set_ylabel('Координата X (м)')
    ax1.set_xlabel('Индекс точки пути')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Зависимость X от индекса точки')

    # График 2: Y от индекса
    ax2.plot([p[1] for p in original_path], label='Исходный путь (Raw)', linestyle='--', color='gray', marker='.', markersize=4)
    ax2.plot([p[1] for p in smoothed_path], label='Сглаженный путь (Spline)', linestyle='-', color='orange', linewidth=2, marker='o', markersize=3)
    #ax2.plot([p[1] for p in actual_path], label='Фактическая траектория (Robot)', linestyle='-', color='blue', alpha=0.7)
    ax2.set_ylabel('Координата Y (м)')
    ax2.set_xlabel('Индекс точки пути')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Зависимость Y от индекса точки')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[INFO] Графики траектории сохранены: {filename}")

def main():
    global target_m, current_path, original_path, smoothed_path_snapshot, actual_path, sock, metrics_printed
    
    print("[INFO] Инициализация системы (Лабораторная 4 - Сглаживание сплайном)...")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.yaml")
    
    try:
        cfg = load_experiment_config(config_path)
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки конфигурации: {e}")
        sys.exit(1)

    robot = cfg["robot"]
    nav = cfg["navigation"]
    pp = cfg["path_planner"]

    planner_config = {
        "type": pp["type"],
        "cell_size": pp["cell_size"],
        "safety_margin": pp["safety_margin"],
        "obs_radius_m": pp["obs_radius_m"],
        "rrt_star": pp["rrt_star"],
        "apf": pp["apf"]
    }

    USE_SPLINE = pp.get("use_spline", True)
    SPLINE_POINTS = pp.get("spline_num_points", 50)

    print("=" * 50)
    print(f"АЛГОРИТМ: {planner_config['type'].upper()}")
    print(f"СПЛАЙН: {'ВКЛЮЧЁН' if USE_SPLINE else 'ВЫКЛЮЧЁН'} (Точек: {SPLINE_POINTS})")
    print("=" * 50)

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

    window_name = "Field & Objects (Lab 4: Spline)"
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

        # 1. Детектирование
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

        # 2. Построение пути (только один раз при появлении цели)
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
                    original_path = new_path # Сохраняем "сырой" путь
                    
                    # Применяем сглаживание и сохраняем снимок для графиков
                    if USE_SPLINE and len(new_path) >= 3:
                        current_path = smooth_path_spline(new_path, num_points=SPLINE_POINTS)
                        smoothed_path_snapshot = list(current_path) # ВАЖНО: сохраняем копию!
                        print(f"[INFO] Путь сглажен: {len(original_path)} -> {len(current_path)} точек")
                    else:
                        current_path = new_path
                        smoothed_path_snapshot = list(new_path)
                else:
                    print("[ERROR] Путь не построен! Робот остановлен.")
            else:
                current_path = []

        # 3. Следование по пути и сбор фактической траектории
        Vx_robot, Vy_robot = 0.0, 0.0
        if current_path and robot_data:
            # Записываем фактическую траекторию (с прореживанием, чтобы не дублировать точки)
            current_pos = robot_data["center_m"]
            if not actual_path or math.hypot(current_pos[0] - actual_path[-1][0], current_pos[1] - actual_path[-1][1]) > 0.01:
                actual_path.append(current_pos)

            next_wp = current_path[0]
            dist_to_wp = math.hypot(next_wp[0] - current_pos[0], next_wp[1] - current_pos[1])

            if dist_to_wp < pp["waypoint_threshold"]:
                current_path.pop(0)
                
                # ================= МАРШРУТ ЗАВЕРШЕН =================
                if not current_path and not metrics_printed:
                    print("[INFO] Цель достигнута!")
                    metrics_printed = True # Блокируем повторный срабатывание
                    
                    # Проверяем наличие данных для расчёта
                    if actual_path and smoothed_path_snapshot and original_path:
                        # Сравниваем фактическую траекторию со сглаженным планом
                        actual_mm, planned_mm, mse, r2 = calculate_tracking_metrics(actual_path, smoothed_path_snapshot)
                        
                        if actual_mm is not None:
                            print("-" * 50)
                            print("[МЕТРИКИ КАЧЕСТВА СЛЕДОВАНИЯ]")
                            print(f"  Планируемая длина: {planned_mm:7.1f} мм")
                            print(f"  Фактический путь:  {actual_mm:7.1f} мм")
                            print(f"  MSE (отклонение):  {mse:7.2f} мм²")
                            print(f"  R² (точность):     {r2:7.4f}")
                            print("-" * 50)
                            
                            # Построение и сохранение графиков (передаем все три пути)
                            save_trajectory_plots(actual_path, smoothed_path_snapshot, original_path, planner_config['type'])
                        else:
                            print("[WARNING] Не удалось рассчитать метрики (недостаточно точек в траектории).")
                    else:
                        print(f"[WARNING] Невозможно рассчитать метрики: actual_path={len(actual_path)}, smoothed={len(smoothed_path_snapshot)}, original={len(original_path)}")
            else:
                Vx_robot, Vy_robot = calculate_robot_velocities(robot_data, next_wp, nav)

        # 4. Отправка команд
        if sock:
            send_velocity(Vx_robot, Vy_robot, 0.0, robot["ip"])

        # 5. Отрисовка
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
        
        # Передаем ОБА пути в отрисовку (original_path отрисуется тонкой линией, current_path - толстой)
        draw_overlay(
            frame, calib, obstacles, robot_data, target_px, nav_data, target_m, 
            current_path=current_path, 
            safety_margin_m=pp["safety_margin"],
            original_path=original_path
        )

        cv2.imshow(window_name, frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
    if sock:
        stop_robot(robot["ip"])
    print("[INFO] Программа завершена.")

if __name__ == "__main__":
    main()