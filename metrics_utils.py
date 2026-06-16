import os
import math
import yaml
import cv2
from datetime import datetime

def calculate_path_length(path):
    """
    Вычисляет длину ломаной линии по списку точек (x, y) в метрах.
    """
    if not path or len(path) < 2:
        return 0.0
    length = sum(math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) for i in range(len(path) - 1))
    return round(length, 3)

def calculate_tracking_metrics(actual_path, planned_path):
    """
    Вычисляет метрики качества следования траектории.
    Возвращает кортеж: (фактический путь в мм, планируемый путь в мм, MSE в мм², R²)
    """
    if not actual_path or not planned_path or len(actual_path) < 2 or len(planned_path) < 2:
        return None, None, None, None
    
    # 1. Фактический пройденный путь
    actual_length_m = sum(math.hypot(actual_path[i+1][0] - actual_path[i][0], 
                                     actual_path[i+1][1] - actual_path[i][1]) 
                          for i in range(len(actual_path) - 1))
    actual_length_mm = actual_length_m * 1000
    
    # 2. Планируемая длина пути
    planned_length_m = sum(math.hypot(planned_path[i+1][0] - planned_path[i][0], 
                                      planned_path[i+1][1] - planned_path[i][1]) 
                           for i in range(len(planned_path) - 1))
    planned_length_mm = planned_length_m * 1000
    
    # 3. MSE (среднеквадратичное отклонение в мм²)
    deviations_mm = []
    for ax, ay in actual_path:
        min_dist = min(math.hypot(ax - px, ay - py) for px, py in planned_path)
        deviations_mm.append(min_dist * 1000)
    
    mse = sum(d**2 for d in deviations_mm) / len(deviations_mm)
    
    # 4. R² (коэффициент детерминации)
    ss_res = sum(d**2 for d in deviations_mm)
    
    mean_x = sum(p[0] for p in planned_path) / len(planned_path)
    mean_y = sum(p[1] for p in planned_path) / len(planned_path)
    
    deviations_from_mean_mm = [math.hypot(ax - mean_x, ay - mean_y) * 1000 for ax, ay in actual_path]
    ss_tot = sum(d**2 for d in deviations_from_mean_mm)
    
    r_squared = 1.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)
    
    return actual_length_mm, planned_length_mm, mse, r_squared

def save_screenshot(frame, algo_name, direction, p1, p2):
    """
    Сохраняет кадр с маршрутом в папку screenshots с информативным именем.
    """
    os.makedirs("screenshots", exist_ok=True)
    p1_str = f"{p1[0]:.1f}_{p1[1]:.1f}"
    p2_str = f"{p2[0]:.1f}_{p2[1]:.1f}"
    filename = f"screenshots/{algo_name}_{direction}_{p1_str}_{p2_str}.png"
    cv2.imwrite(filename, frame)
    print(f"[INFO] Скриншот сохранен: {filename}")

def save_measurements_yaml(algo_name, metrics):
    """
    Сохраняет или обновляет файл measurements.yaml с результатами теста.
    metrics: словарь, содержащий ключи 'forward_len', 'actual_len_m', 'forward_time', 'mse_mm2', 'r2'
    """
    meas_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements.yaml")
    
    if os.path.exists(meas_path):
        with open(meas_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Формируем структуру для сохранения
    data[algo_name] = {
        "last_run": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "planned_path_length_m": round(metrics.get("forward_len", 0.0), 3),
            "actual_path_length_m": round(metrics.get("actual_len_m", 0.0), 3),
            "forward_time_s": round(metrics.get("forward_time", 0.0), 2),
            "mse_mm2": round(metrics.get("mse_mm2", 0.0), 2),
            "r2_accuracy": round(metrics.get("r2", 0.0), 4),
            "success": True
        }
    }
    
    # Сохраняем с сохранением формата и комментариев (насколько это позволяет pyyaml)
    with open(meas_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
    print(f"[INFO] Результаты и метрики сохранены в measurements.yaml")