import cv2
import numpy as np
import math
import os
import sys

# 1. ИМПОРТ МОДУЛЬНЫХ ФУНКЦИЙ
from config_loader import load_experiment_config
from vision_func import (
    load_calibration, setup_aruco_detector, detect_obstacles, detect_robot,
    calculate_navigation, draw_overlay, MIN_OBSTACLE_AREA, COLOR_RANGES, draw_reachable_zone
)
from robot_func import connect_to_robotino, send_velocity, stop_robot, calculate_robot_velocities
from path_planner import plan_path, generate_grid_for_debug, smooth_path_spline

# ================= ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ =================
target_px = None
target_m = None
current_path = []
original_path = []
sock = None

def create_mouse_callback(calib):
    def callback(event, x, y, flags, param):
        global target_px, target_m, current_path, original_path
        if event == cv2.EVENT_LBUTTONDOWN:
            if cv2.pointPolygonTest(calib["field_poly"], (x, y), False) < 0:
                print("[WARNING] Клик за пределами поля.")
                return
            target_px = (x, y)
            pt_top = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), calib["M"])
            target_m = (pt_top[0][0][0] / calib["px_x"], pt_top[0][0][1] / calib["px_y"])
            
            # Сброс путей при новой цели
            current_path = []
            original_path = []
            print(f"[INFO] Цель задана: ({target_m[0]:.2f}, {target_m[1]:.2f} м). Ожидание пути...")
    return callback

def is_path_blocked(current_path, obstacles, safety_radius):
    """
    Быстрая проверка, не блокирует ли какое-либо препятствие 
    ближайшие точки текущего маршрута.
    """
    if not current_path:
        return False
    
    # Проверяем только следующие 5 точек пути для экономии ресурсов
    points_to_check = current_path[:5]
    
    for wp_x, wp_y in points_to_check:
        for obs in obstacles:
            dist = math.hypot(wp_x - obs["center_m"][0], wp_y - obs["center_m"][1])
            if dist < safety_radius:
                return True # Препятствие слишком близко к маршруту
                
    return False

def main():
    global target_m, current_path, original_path, sock
    
    print("[INFO] Инициализация системы (Лабораторная 3 - Динамический объезд препятствий)...")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.yaml")
    
    # 2. ЗАГРУЗКА КОНФИГУРАЦИИ ЧЕРЕЗ ЕДИНЫЙ МОДУЛЬ
    try:
        cfg = load_experiment_config(config_path)
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки конфигурации: {e}")
        sys.exit(1)

    robot = cfg["robot"]
    nav = cfg["navigation"]
    pp = cfg["path_planner"]

    # 3. ФОРМИРОВАНИЕ СЛОВАРЯ ДЛЯ ПЛАНИРОВЩИКА
    planner_config = {
        "type": pp["type"],
        "cell_size": pp["cell_size"],
        "safety_margin": pp["safety_margin"],
        "obs_radius_m": pp["obs_radius_m"],
        "rrt_star": pp["rrt_star"],
        "apf": pp["apf"]
    }

    print("=" * 55)
    print(f"АКТИВНЫЙ ПЛАНИРОВЩИК: {planner_config['type'].upper()}")
    print(f"ДИНАМИЧЕСКОЕ ПЕРЕПЛАНИРОВАНИЕ: ВКЛЮЧЕНО")
    print("=" * 55)

    # 4. ИНИЦИАЛИЗАЦИЯ ОБОРУДОВАНИЯ
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

        # 5. ДЕТЕКТИРОВАНИЕ
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

        # 6. ДИНАМИЧЕСКОЕ ПЛАНИРОВАНИЕ
        replan_needed = False
        safety_radius = pp["obs_radius_m"] + pp["safety_margin"]

        # Условие 1: Пути нет, но есть цель и робот
        if not current_path and target_m and robot_data:
            replan_needed = True
        
        # Условие 2: Путь есть, но он заблокирован новым препятствием
        elif current_path and target_m and robot_data:
            if is_path_blocked(current_path, obstacles, safety_radius):
                print("[INFO] Обнаружено препятствие на маршруте. Перестройка...")
                current_path = []
                original_path = []
                replan_needed = True
            
            # Условие 3: Периодическая перестройка (каждые 20 кадров) для адаптации к плавному движению препятствий
            elif frame_count % 20 == 0:
                replan_needed = True
                current_path = []
                original_path = []

        # Непосредственный вызов планировщика
        if replan_needed and target_m and robot_data:
            dist_to_target = math.hypot(target_m[0] - robot_data["center_m"][0], target_m[1] - robot_data["center_m"][1])
            
            if dist_to_target > pp["waypoint_threshold"] * 1.5:
                new_path = plan_path(
                    robot_data["center_m"], target_m,
                    calib["data"]["width_m"], calib["data"]["height_m"],
                    obstacles, planner_config
                )
                
                if new_path is not None:
                    original_path = new_path # Сохраняем "сырой" путь
                    
                    # Применяем сглаживание, если оно включено в конфиге
                    if pp.get("use_spline", False) and len(new_path) >= 3:
                        current_path = smooth_path_spline(new_path, num_points=pp.get("spline_num_points", 50))
                    else:
                        current_path = new_path
                        
                    print(f"[INFO] Путь построен: {len(current_path)} точек")
                else:
                    print("[ERROR] Путь не построен! Робот остановлен.")
                    current_path = []
                    original_path = []
            else:
                current_path = []
                original_path = []

        # 7. СЛЕДОВАНИЕ ПО ПУТИ
        Vx_robot, Vy_robot = 0.0, 0.0
        if current_path and robot_data:
            next_wp = current_path[0]
            dist_to_wp = math.hypot(next_wp[0] - robot_data["center_m"][0], next_wp[1] - robot_data["center_m"][1])

            if dist_to_wp < pp["waypoint_threshold"]:
                current_path.pop(0)
                if not current_path:
                    print("[INFO] Цель достигнута!")
            else:
                # >>> ИСПОЛЬЗОВАНИЕ МОДУЛЬНОЙ ФУНКЦИИ КИНЕМАТИКИ <<<
                Vx_robot, Vy_robot = calculate_robot_velocities(robot_data, next_wp, nav)

        # 8. ОТПРАВКА КОМАНД
        if sock:
            send_velocity(Vx_robot, Vy_robot, 0.0, robot["ip"])

        # 9. ОТРИСОВКА
        grid_current = generate_grid_for_debug(
            calib["data"]["width_m"], calib["data"]["height_m"], 
            obstacles, pp["cell_size"], pp["safety_margin"], pp["obs_radius_m"]
        )
        draw_reachable_zone(frame, grid_current, robot_data["center_m"] if robot_data else None, calib, pp["cell_size"], alpha=0.25)

        nav_data = calculate_navigation(robot_data, target_m) if (robot_data and target_m) else None 
        
        # Безопасный вызов с именованными аргументами
        draw_overlay(
            frame, calib, obstacles, robot_data, target_px, nav_data, target_m, 
            current_path=current_path, 
            safety_margin_m=pp["safety_margin"],
            original_path=original_path
        )

        # Информационный оверлей на кадре
        cv2.putText(frame, f"Algo: {planner_config['type'].upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if current_path:
            # Простой расчет длины для отображения (можно импортировать calculate_path_length, если нужно)
            curr_len = sum(math.hypot(current_path[i+1][0]-current_path[i][0], current_path[i+1][1]-current_path[i][1]) for i in range(len(current_path)-1))
            cv2.putText(frame, f"Path Len: {curr_len:.2f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    # 10. ЗАВЕРШЕНИЕ
    cap.release()
    cv2.destroyAllWindows()
    if sock:
        stop_robot(robot["ip"])
    print("[INFO] Программа завершена.")

if __name__ == "__main__":
    main()