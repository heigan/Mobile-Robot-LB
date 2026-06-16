import cv2
import numpy as np
import os
from ruamel.yaml import YAML
from vision_func import load_calibration

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.yaml")
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 2:
            clicked_points.append((x, y))
            label = "START" if len(clicked_points) == 1 else "GOAL"
            print(f"[INFO] Зафиксирована точка {label}: ({x}, {y})")

def main():
    if not os.path.exists(CONFIG_PATH):
        print("[ERROR] parameters.yaml не найден.")
        return

    # Инициализируем ruamel.yaml (он сохраняет комментарии!)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2) # Красивые отступы

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.load(f)
        if config is None:
            config = {}

    calib = load_calibration(CONFIG_PATH, 1000, 800)
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("[ERROR] Камера не открыта.")
        return

    cv2.namedWindow("Setup Test Points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Setup Test Points", mouse_callback)
    print("="*50)
    print("НАСТРОЙКА ТЕСТОВОГО МАРШРУТА")
    print("1. Кликните ЛКМ для установки ТОЧКИ СТАРТА.")
    print("2. Кликните ЛКМ для установки ТОЧКИ ЦЕЛИ.")
    print("3. Нажмите 'q' или ESC для сохранения и выхода.")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret: break

        display = frame.copy()
        for i, (px, py) in enumerate(clicked_points):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            label = "START" if i == 0 else "GOAL"
            cv2.circle(display, (px, py), 6, color, -1)
            cv2.putText(display, label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Setup Test Points", display)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(clicked_points) == 2:
        # Перевод пикселей в метры с округлением до 2 знаков
        def px_to_m(px, py):
            pt_top = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), calib["M"])
            x_m = round(float(pt_top[0][0][0] / calib["px_x"]), 2)
            y_m = round(float(pt_top[0][0][1] / calib["px_y"]), 2)
            return x_m, y_m

        start_m = px_to_m(*clicked_points[0])
        goal_m = px_to_m(*clicked_points[1])

        # Создаем или обновляем секцию, не трогая остальные
        if "recorded_path_test" not in config:
            config["recorded_path_test"] = {}
            
        config["recorded_path_test"]["enabled"] = True
        
        # Используем ruamel.yaml.comment.CommentedSeq, чтобы сохранить формат [x, y] в одну строку
        from ruamel.yaml.comments import CommentedSeq
        config["recorded_path_test"]["start_m"] = CommentedSeq([start_m[0], start_m[1]])
        config["recorded_path_test"]["goal_m"] = CommentedSeq([goal_m[0], goal_m[1]])
        config["recorded_path_test"]["start_m"].fa.set_flow_style() # Принудительно делаем [x, y]
        config["recorded_path_test"]["goal_m"].fa.set_flow_style()

        # Сохраняем обратно (комментарии останутся на месте!)
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        print(f"\nУспешно и аккуратно сохранено в {CONFIG_PATH}:")
        print(f"   Start: {start_m} м")
        print(f"   Goal : {goal_m} м")
        print("Теперь запустите Lb_2.py для проведения теста.")
    else:
        print("\nСохранение отменено: выбрано менее 2 точек.")

if __name__ == "__main__":
    main()