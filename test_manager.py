import time
from datetime import datetime
import os
import yaml
from path_planner import calculate_tracking_metrics

class AutoTestManager:
    def __init__(self, start_m: tuple, goal_m: tuple, algo_name: str):
        self.start_m = start_m
        self.goal_m = goal_m
        self.algo_name = algo_name
        
        # Начинаем сразу с движения к цели
        self.state = "MEASURE_FORWARD"  
        self.current_target = goal_m  
        
        self.metrics = {
            "forward_len": 0.0, "forward_time": 0.0,
            "reverse_len": 0.0, "reverse_time": 0.0,
            "actual_len_m": 0.0, "mse_mm2": 0.0, "r2": 0.0, "planned_len_m": 0.0
        }
        self.screenshot_pending = None

    def get_current_target(self) -> tuple:
        return self.current_target

    def on_path_built(self, path_length_m: float):
        """Вызывается сразу после успешного построения нового пути."""
        if self.state == "MEASURE_FORWARD":
            self.metrics["forward_len"] = path_length_m
            self.screenshot_pending = "forward"  # Запланировать скриншот
        elif self.state == "MEASURE_REVERSE":
            self.metrics["reverse_len"] = path_length_m
            self.screenshot_pending = "reverse"  # Запланировать скриншот

    def on_waypoint_reached(self, elapsed_time: float, actual_path: list, planned_path: list) -> dict:
        """Вызывается, когда робот достиг конца текущего пути."""
        actions = {
            "save_screenshot": False, "print_metrics": False, 
            "next_state": self.state, "reset_paths": False, "save_final_yaml": False,
            "metrics_data": None
        }
        
        # Расчет метрик
        if actual_path and planned_path:
            actual_mm, planned_mm, mse, r2 = calculate_tracking_metrics(actual_path, planned_path)
            if actual_mm is not None:
                self.metrics.update({
                    "actual_len_m": actual_mm / 1000.0,
                    "mse_mm2": mse,
                    "r2": r2,
                    "planned_len_m": planned_mm / 1000.0
                })
                actions["print_metrics"] = True
                actions["metrics_data"] = (actual_mm, planned_mm, mse, r2)

        # Упрощённая логика переключения состояний
        if self.state == "MEASURE_FORWARD":
            self.metrics["forward_time"] = elapsed_time
            print(f"[INFO] Замер 'Туда' завершён. Время: {elapsed_time:.2f} с. Начинаем возврат.")
            self.state = "MEASURE_REVERSE"
            self.current_target = self.start_m
            actions["next_state"] = self.state
            actions["reset_paths"] = True

        elif self.state == "MEASURE_REVERSE":
            self.metrics["reverse_time"] = elapsed_time
            print("[INFO] Тест завершён! Возврат на старт выполнен.")
            self.state = "FINISHED"
            actions["next_state"] = self.state
            actions["save_final_yaml"] = True

        return actions

    def is_finished(self) -> bool:
        return self.state == "FINISHED"

    def get_screenshot_task(self):
        task = self.screenshot_pending
        self.screenshot_pending = None
        return task

    def save_final_report(self):
        meas_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements.yaml")
        if os.path.exists(meas_path):
            with open(meas_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        data[self.algo_name] = {
            "last_run": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "planned_path_length_m": round(self.metrics.get("planned_len_m", 0.0), 3),
                "actual_path_length_m": round(self.metrics.get("actual_len_m", 0.0), 3),
                "forward_time_s": round(self.metrics.get("forward_time", 0.0), 2),
                "mse_mm2": round(self.metrics.get("mse_mm2", 0.0), 2),
                "r2_accuracy": round(self.metrics.get("r2", 0.0), 4),
                "success": True
            }
        }
        with open(meas_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print("[INFO] Результаты и метрики сохранены в measurements.yaml")