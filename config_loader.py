import os
import yaml

def load_experiment_config(config_path: str) -> dict:
    """
    Загружает и валидирует конфигурацию из parameters.yaml.
    Возвращает структурированный словарь с гарантированными значениями по умолчанию.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        full_cfg = yaml.safe_load(f) or {}

    # 1. Конфигурация робота
    robot_cfg = full_cfg.get("robot", {})
    robot = {
        "ip": robot_cfg.get("robot_id", "192.168.0.1"),
        "port": robot_cfg.get("robot_port", 80),
        "diameter_m": robot_cfg.get("robot_diameter_m", 0.45)
    }

    # 2. Конфигурация навигации
    nav_cfg = full_cfg.get("navigation", {})
    navigation = {
        "speed_far_m_s": nav_cfg.get("speed_far_m_s", 0.5),
        "speed_near_m_s": nav_cfg.get("speed_near_m_s", 0.05),
        "speed_threshold_m": nav_cfg.get("speed_threshold_m", 0.3)
    }

    # 3. Конфигурация планировщика
    pp_cfg = full_cfg.get("path_planner", {})
    path_planner = {
        "type": pp_cfg.get("planner_type", "astar").lower(),
        "cell_size": pp_cfg.get("cell_size", 0.05),
        "safety_margin": pp_cfg.get("safety_margin", 0.05),
        "obs_radius_m": pp_cfg.get("obs_radius_m", robot["diameter_m"] / 2),
        "waypoint_threshold": pp_cfg.get("waypoint_threshold", 0.15),
        "use_spline": pp_cfg.get("use_spline", False),
        "spline_num_points": pp_cfg.get("spline_num_points", 50),
        "rrt_star": pp_cfg.get("rrt_star", {}),
        "apf": pp_cfg.get("apf", {})
    }

    # 4. Конфигурация автоматического теста
    test_cfg = full_cfg.get("recorded_path_test", {})
    recorded_test = {
        "enabled": test_cfg.get("enabled", False),
        "start_m": tuple(test_cfg.get("start_m", [0.5, 0.5])),
        "goal_m": tuple(test_cfg.get("goal_m", [1.5, 1.5]))
    }

    return {
        "robot": robot,
        "navigation": navigation,
        "path_planner": path_planner,
        "recorded_test": recorded_test
    }