import cv2
import numpy as np
import yaml
import os
from vision_func import (
    load_calibration, setup_aruco_detector, detect_obstacles,
    detect_robot, calculate_navigation, draw_overlay
)

# Настройки путей и окна
VIDEO_SOURCE = 1  # Индекс камеры (0 - встроенная, 1 - USB, или путь к файлу)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.yaml")
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

clicked_points = []
selection_complete = False

def mouse_callback(event, x, y, flags, param):
    global clicked_points, selection_complete
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((int(x), int(y)))
            print(f"Зафиксирована точка {len(clicked_points)}: ({x}, {y})")
        if len(clicked_points) == 4:
            selection_complete = True

# 1. Открытие источника видео
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Ошибка: не удалось открыть видео/камеру.")
    exit()

# Захват первого кадра для калибровки
ret, frame = cap.read()
if not ret:
    print("Ошибка: не удалось прочитать кадр.")
    exit()

# Настройка окна
cv2.namedWindow("Select 4 Corners", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select 4 Corners", WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.setMouseCallback("Select 4 Corners", mouse_callback)

print("="*40)
print("КАЛИБРОВКА ПОЛЯ")
print("="*40)
print("1. Кликните по 4 углам белого поля (по порядку).")
print("2. Первая точка станет началом координат (0,0).")
print("3. Нажмите ESC для отмены.")

while not selection_complete:
    display_frame = frame.copy()
    
    # Отрисовка точек и линий
    for i, (px, py) in enumerate(clicked_points):
        cv2.circle(display_frame, (px, py), 4, (0, 0, 255), -1)
        cv2.putText(display_frame, f"P{i+1}", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    if len(clicked_points) >= 2:
        for i in range(len(clicked_points) - 1):
            cv2.line(display_frame, clicked_points[i], clicked_points[i + 1], (0, 255, 0), 2)

    cv2.imshow("Select 4 Corners", display_frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        print("Программа остановлена.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# 2. Ввод реальных размеров
print("\nВведите реальные размеры поля (в метрах):")
try:
    real_width_m = float(input("Ширина (вдоль оси X): "))
    real_height_m = float(input("Высота (вдоль оси Y): "))
    robot_diameter_m = float(input("Диаметр робота: "))
except ValueError:
    print("Ошибка: введите корректные числа.")
    exit()

# 3. Сохранение в YAML (обновляет только секции field и robot)
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
else:
    config = {}

config["field"] = {
    "corners": [list(pt) for pt in clicked_points],
    "origin": list(clicked_points[0]),
    "width_m": real_width_m,
    "height_m": real_height_m
}

if "robot" not in config:
    config["robot"] = {}
config["robot"]["robot_diameter_m"] = robot_diameter_m
config["robot"]["robot_id"] = config["robot"].get("robot_id", "192.168.0.1")
config["robot"]["robot_port"] = config["robot"].get("robot_port", 80)

# Сохраняем остальные секции (navigation, path_planner) без изменений
with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f"Конфигурация обновлена в {CONFIG_PATH}")

# Записываем обратно
with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
print(f"Конфигурация обновлена в {CONFIG_PATH}")

# 4. Проверка: отрисовка границ и осей на кадре
cv2.namedWindow("Field Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Field Preview", WINDOW_WIDTH, WINDOW_HEIGHT)

pts = np.array(clicked_points, np.int32).reshape((-1, 1, 2))
origin = clicked_points[0]
vec_x = np.array(clicked_points[1]) - np.array(origin)
vec_y = np.array(clicked_points[3]) - np.array(origin)
axis_scale = 0.1

axis_x_end = tuple((np.array(origin) + vec_x * axis_scale).astype(int))
axis_y_end = tuple((np.array(origin) + vec_y * axis_scale).astype(int))

print("\nОкно проверки. Нажмите 'q' или ESC для выхода...")

while True:
    # Если видео с камеры - читаем новый кадр, если файл - можно закольцевать
    if isinstance(VIDEO_SOURCE, int):
        ret, frame = cap.read()
        if not ret:
            print("Видео/поток завершён.")
            break
    else:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

    # Отрисовка
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    
    # Ось X (синяя)
    cv2.arrowedLine(frame, origin, axis_x_end, (255, 0, 0), 1, tipLength=0.2)
    cv2.putText(frame, "X ", (axis_x_end[0] + 10, axis_x_end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Ось Y (красная)
    cv2.arrowedLine(frame, origin, axis_y_end, (0, 0, 255), 1, tipLength=0.2)
    cv2.putText(frame, "Y ", (axis_y_end[0] + 5, axis_y_end[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Начало координат
    cv2.circle(frame, origin, 5, (0, 0, 255), -1)
    cv2.putText(frame, "0 ", (origin[0] - 15, origin[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Field Preview", frame)
    key = cv2.waitKey(30)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Программа завершена.")