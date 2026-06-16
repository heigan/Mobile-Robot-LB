import cv2
import numpy as np
import yaml
import os

# ================= НАСТРОЙКИ =================
CAMERA_INDEX = 1  # 1 - внешняя USB-камера, 0 - встроенная
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.yaml")

WARPED_WIDTH = 800
WARPED_HEIGHT = 600
MIN_OBSTACLE_AREA = 300

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

# ================= 1. ЗАГРУЗКА КОНФИГА (YAML) =================
if not os.path.exists(CONFIG_PATH):
    print("Ошибка: файл parameters.yaml не найден.")
    exit()

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

field = config.get("field", {})
src_pts = np.array(field["corners"], dtype=np.float32).reshape(4, 2)
real_width_m = field.get("width_m")
real_height_m = field.get("height_m")

if real_width_m is None or real_height_m is None:
    print("Ошибка: в YAML отсутствуют field.width_m или field.height_m")
    exit()

# Матрица гомографии и коэффициенты масштабирования
dst_pts = np.array([[0, 0], [WARPED_WIDTH, 0], [WARPED_WIDTH, WARPED_HEIGHT], [0, WARPED_HEIGHT]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
px_per_m_x = WARPED_WIDTH / real_width_m
px_per_m_y = WARPED_HEIGHT / real_height_m

color_ranges = {
    "Red": [[(0, 100, 100), (10, 255, 255)], [(160, 100, 100), (180, 255, 255)]],
    "Green": [[(35, 50, 50), (85, 255, 255)]],
    "Blue": [[(90, 50, 50), (130, 255, 255)]],
    "Black": [[(0, 0, 0), (180, 255, 110)]]
}

# ================= 2. ПОДКЛЮЧЕНИЕ КАМЕРЫ =================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Убирает накопление кадров и снижает лаги
if not cap.isOpened():
    print(f"Ошибка: не удалось открыть камеру (индекс {CAMERA_INDEX}). Попробуйте 0 или 2.")
    exit()

cv2.namedWindow("Obstacles", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Obstacles", 900, 700)
print("Определение препятствий. Нажмите 'q' или ESC для выхода.\n")

# Подготовка констант для отрисовки
corners_int = np.array(field["corners"], dtype=np.int32).reshape(4, 2)
origin_pt = tuple(corners_int[0])
vec_x = corners_int[1] - corners_int[0]
vec_y = corners_int[3] - corners_int[0]
axis_scale = 0.1
axis_x_end = tuple((corners_int[0] + vec_x * axis_scale).astype(int))
axis_y_end = tuple((corners_int[0] + vec_y * axis_scale).astype(int))
field_poly = src_pts.reshape(-1, 1, 2)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Потеря кадра. Проверьте подключение камеры.")
        continue

    # Отрисовка границ и осей координат
    cv2.polylines(frame, [corners_int.reshape(-1, 1, 2)], True, GREEN, 3)
    cv2.arrowedLine(frame, origin_pt, axis_x_end, BLUE, 2, tipLength=0.2)
    cv2.putText(frame, "X", (axis_x_end[0] + 10, axis_x_end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2)
    cv2.arrowedLine(frame, origin_pt, axis_y_end, RED, 2, tipLength=0.2)
    cv2.putText(frame, "Y", (axis_y_end[0] + 5, axis_y_end[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    cv2.circle(frame, origin_pt, 5, RED, -1)
    cv2.putText(frame, "0", (origin_pt[0] - 15, origin_pt[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

    # Поиск препятствий
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    obstacles = []
    
    for color_name, ranges in color_ranges.items():
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for r_min, r_max in ranges:
            mask |= cv2.inRange(hsv, np.array(r_min, dtype=np.uint8), np.array(r_max, dtype=np.uint8))


        # --- НАЛОЖЕНИЕ МАСКИ ЧЕРНЫХ ПРЕПЯТСТВИЙ НА ВИДЕО ---
        if color_name == "Black":
            overlay = frame.copy()
            overlay[mask > 0] = (0, 255, 0)  # Красные пиксели там, где сработала маска
            frame = cv2.addWeighted(overlay, 1, frame, 0.0, 0)
        # ---------------------------------------------------

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) <= MIN_OBSTACLE_AREA:
                continue

            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx_orig = int(m["m10"] / m["m00"])
            cy_orig = int(m["m01"] / m["m00"])

            if cv2.pointPolygonTest(field_poly, (cx_orig, cy_orig), False) < 0:
                continue

            pt_top = cv2.perspectiveTransform(np.array([[[cx_orig, cy_orig]]], dtype=np.float32), M)
            cx_t, cy_t = pt_top[0][0]
            cx_m = cx_t / px_per_m_x
            cy_m = cy_t / px_per_m_y

            obstacles.append({
                "color": color_name,
                "center_m": (round(cx_m, 2), round(cy_m, 2))
            })

            cv2.circle(frame, (cx_orig, cy_orig), 6, GREEN, -1)
            cv2.putText(frame, f"{color_name} ({cx_m:.2f}, {cy_m:.2f}m)",
                        (cx_orig + 10, cy_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

    if frame_count % 30 == 0 and obstacles:
        print(f"--- Кадр {frame_count} ---")
        for obs in obstacles:
            print(f"  {obs['color']}: центр=({obs['center_m'][0]} м, {obs['center_m'][1]} м)")
        print()

    cv2.imshow("Obstacles", frame)
    frame_count += 1

    key = cv2.waitKey(30)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Программа завершена.")