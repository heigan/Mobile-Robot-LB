import cv2
import numpy as np
import math
import os
import yaml
from collections import deque


# Цвета и константы
COLORS = {"GREEN": (0, 255, 0), "BLUE": (255, 0, 0), "RED": (0, 0, 255), 
          "PURPLE": (255, 0, 255), "YELLOW": (0, 255, 255), "Black": (0, 0, 0)}
MIN_OBSTACLE_AREA = 300
COLOR_RANGES = {
    "Red": [[(0, 100, 100), (10, 255, 255)], [(160, 100, 100), (180, 255, 255)]],
    "Green": [[(35, 50, 50), (85, 255, 255)]],
    "Blue": [[(90, 50, 50), (130, 255, 255)]],
    "Black": [[(0, 0, 0), (180, 255, 105)]]  
}


def load_calibration(config_path: str, warped_w: int, warped_h: int) -> dict:
    """
    Загружает параметры поля из YAML, строит матрицы перспективного преобразования
    и коэффициенты масштабирования.
    
    Входы:
        config_path: путь к parameters.yaml
        warped_w, warped_h: размеры виртуального вида сверху (в пикселях)
    Выходы:
        dict: словарь с матрицами M, M_inv, коэффициентами px_x/px_y, 
              углами поля corners_int и полигоном field_poly
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    field = config.get('field', {})
    required_keys = ['corners', 'width_m', 'height_m']
    if not all(k in field for k in required_keys):
        raise ValueError(f"В секции 'field' конфигурации отсутствуют обязательные поля: {required_keys}")

    # Обработка углов: поддерживает как плоский список [x1,y1,x2,y2...], так и [[x1,y1],...]
    src_pts = np.array(field['corners'], dtype=np.float32)
    if src_pts.shape[0] != 4:
        src_pts = src_pts.reshape(4, 2)
        
    dst_pts = np.array([[0, 0], [warped_w, 0], [warped_w, warped_h], [0, warped_h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = np.linalg.inv(M)
    px_per_m_x = warped_w / field['width_m']
    px_per_m_y = warped_h / field['height_m']

    return {
        "data": field,
        "M": M, 
        "M_inv": M_inv,
        "px_x": px_per_m_x, 
        "px_y": px_per_m_y,
        "corners_int": np.array(field['corners'], dtype=np.int32).reshape(4, 2),
        "field_poly": src_pts.reshape(-1, 1, 2)
    }
pass

def setup_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
    return cv2.aruco.ArucoDetector(aruco_dict, params)
pass

def detect_obstacles(hsv_frame, M, px_x, px_y, field_poly, min_area=MIN_OBSTACLE_AREA, 
                     color_ranges=COLOR_RANGES, aruco_center=None, aruco_radius=0):
    obstacles = []
    h, w = hsv_frame.shape[:2]

    for color_name, ranges in color_ranges.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        for r_min, r_max in ranges:
            mask |= cv2.inRange(hsv_frame, np.array(r_min, dtype=np.uint8), np.array(r_max, dtype=np.uint8))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

        # ИГНОРИРОВАНИЕ ARUCO: рисуем чёрный круг (0) на маске, "стирая" маркер
        if color_name == "Black" and aruco_center is not None:
            cv2.circle(mask, aruco_center, int(aruco_radius * 1.1), 0, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) <= min_area:
                continue

            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx_orig, cy_orig = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

            if cv2.pointPolygonTest(field_poly, (cx_orig, cy_orig), False) < 0:
                continue

            pt_top = cv2.perspectiveTransform(np.array([[[cx_orig, cy_orig]]], dtype=np.float32), M)
            cx_m = pt_top[0][0][0] / px_x
            cy_m = pt_top[0][0][1] / px_y
            obstacles.append({
                "color": color_name, 
                "center_m": (round(cx_m, 2), round(cy_m, 2)), 
                "center_px": (cx_orig, cy_orig),
                "contour": cnt
            })

    return obstacles
pass

def detect_robot(gray_frame, detector, M, M_inv, px_x, px_y, robot_diameter_m, field_poly):
    # Находит ArUco-маркер, возвращает данные робота или None.
    corners, ids, _ = detector.detectMarkers(gray_frame)
    if ids is None or len(ids) == 0:
        return None

    pts = corners[0][0]
    cx_orig, cy_orig = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))

    if cv2.pointPolygonTest(field_poly, (cx_orig, cy_orig), False) < 0:
        return None

    pt_top = cv2.perspectiveTransform(np.array([[[cx_orig, cy_orig]]], dtype=np.float32), M)
    cx_t, cy_t = pt_top[0][0]
    cx_m, cy_m = cx_t / px_x, cy_t / px_y

    radius_m = robot_diameter_m / 2.0
    center_orig = cv2.perspectiveTransform(np.array([[[cx_t, cy_t]]], dtype=np.float32), M_inv)[0][0]
    edge_orig = cv2.perspectiveTransform(np.array([[[cx_t + radius_m * px_x, cy_t]]], dtype=np.float32), M_inv)[0][0]
    pixel_radius = int(np.linalg.norm(edge_orig - center_orig))

    return {
        "id": int(ids[0][0]),
        "center_m": (round(cx_m, 2), round(cy_m, 2)),
        "center_px": (int(center_orig[0]), int(center_orig[1])),
        "radius_px": pixel_radius,
        "marker_pts": pts, "pts_top": cv2.perspectiveTransform(pts.reshape(1, -1, 2), M)[0]
    }
pass

def calculate_navigation(robot_data, target_m):
    # Вычисляет дистанцию до цели и угол относительно курса робота.
    rx, ry = robot_data["center_m"]
    tx, ty = target_m

    dist_m = math.hypot(tx - rx, ty - ry)

    pts_top = robot_data["pts_top"]
    robot_heading = math.atan2(pts_top[1][1] - pts_top[0][1], pts_top[1][0] - pts_top[0][0])
    target_heading = math.atan2(ty - ry, tx - rx)

    angle_diff = target_heading - robot_heading
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
    return dist_m, math.degrees(angle_diff)
pass

def draw_forbidden_zone(frame, calib, obstacles, safety_margin_m, alpha=0.35):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    corners = calib["corners_int"]

    # 1. Перевод метров в пиксели (консервативная оценка)
    px_per_m = min(calib["px_x"], calib["px_y"])
    safe_px = int(safety_margin_m * px_per_m)

    # Жёсткая защита: зона не может быть шире 120px, даже при ошибке в YAML
    safe_px = np.clip(safe_px, 3, 120)

    # 2. Рисуем контур поля и заполняем препятствия
    cv2.polylines(mask, [corners.reshape(-1, 1, 2)], True, 255, 1)
    for obs in obstacles:
        if "contour" in obs:
            cv2.fillPoly(mask, [obs["contour"]], 255)

    # 3. Расширяем маску. Ядро (2*R+1) даёт расширение ровно на R пикселей во все стороны
    if safe_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_px*2+1, safe_px*2+1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # 4. Полупрозрачное наложение
    overlay = frame.copy()
    overlay[mask > 0] = (0, 0, 180)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
pass

def draw_reachable_zone(frame, grid, robot_m, calib, cell_size, alpha=0.25):
    """
    Отрисовывает зелёную полупрозрачную зону, куда можно поставить цель, 
    и маршрут A* гарантированно будет построен.
    """
    if robot_m is None or grid is None:
        return

    rows, cols = grid.shape
    start_c = int(robot_m[0] / cell_size)
    start_r = int(robot_m[1] / cell_size)

    # Если робот в заблокированной ячейке, зона пуста
    if not (0 <= start_c < cols and 0 <= start_r < rows) or grid[start_r, start_c] == 1:
        return
    
    grid_work = grid.copy()
    grid_work[start_r, start_c] = 0

    # BFS: поиск всех достижимых свободных ячеек (8-связность, как в A*)
    queue = deque([(start_r, start_c)])
    reachable = np.zeros((rows, cols), dtype=bool)
    reachable[start_r, start_c] = True

    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and grid_work[nr, nc] == 0:
                    reachable[nr, nc] = True
                    queue.append((nr, nc))

    # Векторизованная проекция достижимых ячеек на исходный кадр
    r_idx, c_idx = np.where(reachable)
    if len(r_idx) == 0: return

    cx_m = (c_idx + 0.5) * cell_size
    cy_m = (r_idx + 0.5) * cell_size
    pts_top = np.stack([cx_m * calib["px_x"], cy_m * calib["px_y"]], axis=-1).reshape(-1, 1, 2).astype(np.float32)
    pts_orig = cv2.perspectiveTransform(pts_top, calib["M_inv"])

    # Наложение зелёного слоя
    overlay = frame.copy()
    for pt in pts_orig:
        cv2.circle(overlay, (int(pt[0,0]), int(pt[0,1])), 4, (0, 255, 0), -1)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
pass

def draw_overlay(frame, calib, obstacles, robot_data, target_px, nav_data, target_m, current_path=None, safety_margin_m=0.05, original_path=None):
    # Отрисовка запрещенной зоны
    # draw_forbidden_zone(frame, calib, obstacles, safety_margin_m, alpha=0.35)
    
    # Оси координат
    corners = calib["corners_int"]
    origin = tuple(corners[0])
    vec_x = corners[1] - corners[0]
    vec_y = corners[3] - corners[0]
    axis_scale = 0.1
    ax_end = tuple((corners[0] + vec_x * axis_scale).astype(int))
    ay_end = tuple((corners[0] + vec_y * axis_scale).astype(int))

    cv2.arrowedLine(frame, origin, ax_end, COLORS["BLUE"], 2, tipLength=0.2)
    cv2.putText(frame, "X", (ax_end[0] + 10, ax_end[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["BLUE"], 2)
    cv2.arrowedLine(frame, origin, ay_end, COLORS["RED"], 2, tipLength=0.2)
    cv2.putText(frame, "Y", (ay_end[0] + 5, ay_end[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["RED"], 2)
    cv2.circle(frame, origin, 5, COLORS["RED"], -1)
    cv2.putText(frame, "0", (origin[0] - 15, origin[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["RED"], 2)

    # Робот
    if robot_data:
        cv2.circle(frame, robot_data["center_px"], robot_data["radius_px"], COLORS["Black"], 2)
        p0 = robot_data["marker_pts"][0].astype(np.float32)
        p1 = robot_data["marker_pts"][1].astype(np.float32)
        p3 = robot_data["marker_pts"][3].astype(np.float32)
        center = np.array(robot_data["center_px"], dtype=np.float32)
        axis_len = 45

        vec_x = p1 - p0
        norm_x = np.linalg.norm(vec_x)
        dir_x = vec_x / norm_x if norm_x > 0 else np.array([1.0, 0.0])
        x_end = center + dir_x * axis_len
        cv2.arrowedLine(frame, tuple(center.astype(int)), tuple(x_end.astype(int)), COLORS["Black"], 2, tipLength=0.2)
        cv2.putText(frame, "Y_r", tuple(x_end.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["Black"], 1)

        vec_y = p3 - p0
        norm_y = np.linalg.norm(vec_y)
        dir_y = vec_y / norm_y if norm_y > 0 else np.array([0.0, 1.0])
        y_end = center + dir_y * axis_len
        cv2.arrowedLine(frame, tuple(center.astype(int)), tuple(y_end.astype(int)), COLORS["Black"], 2, tipLength=0.2)
        cv2.putText(frame, "X_r", tuple(y_end.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["Black"], 1)

    # ОТРИСОВКА ТЕКУЩЕГО (СГЛАЖЕННОГО) ПУТИ 
    if current_path and len(current_path) > 0 and robot_data:
        path_pixels = []
        for px_m, py_m in current_path:
            pt_top = np.array([[[px_m * calib["px_x"], py_m * calib["px_y"]]]], dtype=np.float32)
            pt_orig = cv2.perspectiveTransform(pt_top, calib["M_inv"])[0][0]
            path_pixels.append((int(pt_orig[0]), int(pt_orig[1])))
        
        if len(path_pixels) > 1:
            for i in range(len(path_pixels) - 1):
                cv2.line(frame, path_pixels[i], path_pixels[i+1], COLORS["YELLOW"], 2)
            cv2.circle(frame, path_pixels[-1], 5, COLORS["YELLOW"], -1)
            
    elif target_px and not current_path:
        cv2.circle(frame, target_px, 4, COLORS["YELLOW"], -1)
        if robot_data:
            cv2.line(frame, robot_data["center_px"], target_px, COLORS["YELLOW"], 2)
    
    # ОТРИСОВКА ИСХОДНОГО (СЫРОГО) ПУТИ
    if original_path is not None and len(original_path) > 1 and robot_data:
        path_pixels_orig = []
        for px_m, py_m in original_path:
            pt_top = np.array([[[px_m * calib["px_x"], py_m * calib["px_y"]]]], dtype=np.float32)
            pt_orig = cv2.perspectiveTransform(pt_top, calib["M_inv"])[0][0]
            path_pixels_orig.append((int(pt_orig[0]), int(pt_orig[1])))
        
        if len(path_pixels_orig) > 1:
            for i in range(len(path_pixels_orig) - 1):
                cv2.line(frame, path_pixels_orig[i], path_pixels_orig[i+1], COLORS["Black"], 1)               


    # Текст (дистанция и координаты)
    h, w = frame.shape[:2]
    if target_m is not None:
        cv2.putText(frame, f"T({target_m[0]:.2f}, {target_m[1]:.2f}m)", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["YELLOW"], 2)
    if robot_data:
        cv2.putText(frame, f"R{robot_data['id']} ({robot_data['center_m'][0]:.2f}, {robot_data['center_m'][1]:.2f}m)", 
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["YELLOW"], 2)
    if nav_data:
        cv2.putText(frame, f"Dist: {nav_data[0]:.2f}m | Angle: {nav_data[1]:.1f} deg", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["YELLOW"], 2)

pass