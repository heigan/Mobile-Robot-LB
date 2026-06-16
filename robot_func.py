import requests
import socket
import yaml
import os
from typing import Optional, Dict
import math

def load_robot_config(config_path: str = "parameters.yaml") -> Dict:
    """
    Загружает конфигурацию робота из YAML файла.
    Возвращает словарь с параметрами подключения.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def connect_to_robotino(ip: str, port: int) -> Optional[socket.socket]:
    """
    Устанавливает TCP-сокет соединение с контроллером робота.
    Входы:
        ip: str — IP-адрес робота
        port: int — порт TCP-подключения
    Выходы:
        socket.socket | None — объект сокета при успехе, None при ошибке
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)  # Таймаут подключения 5 секунд
        sock.connect((ip, port))
        print(f"TCP соединение установлено: {ip}:{port}")
        return sock
    except Exception as e:
        print(f"Ошибка TCP подключения к {ip}:{port}: {e}")
        return None

def send_velocity(vx: float, vy: float, omega: float, ip: str, timeout: float = 3.0) -> bool:
    """
    Отправляет вектор скорости роботу через HTTP API (omnidrive).
    Входы:
        vx: float — линейная скорость вперёд (м/с)
        vy: float — боковая скорость влево (м/с)
        omega: float — угловая скорость (рад/с)
        ip: str — IP-адрес робота
        timeout: float — таймаут HTTP-запроса
    Выходы:
        bool — True при успешной отправке, False при сетевой или HTTP-ошибке
    """
    url = f"http://{ip}/data/omnidrive"
    payload = [vx, vy, omega]
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code == 200:
            return True
        else:
            print(f"Ошибка HTTP {response.status_code}: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Сетевая ошибка при отправке скорости: {e}")
        return False
    
def calculate_robot_velocities(robot_data: dict, next_wp: tuple, nav_config: dict) -> tuple:
    """
    Вычисляет линейные скорости робота (Vx, Vy) в его локальной системе координат
    для движения к следующей точке пути.
    
    Входы:
        robot_data: словарь с ключами 'center_m' (x, y в метрах) и 'pts_top' (точки маркера)
        next_wp: кортеж (x, y) целевой точки пути в метрах
        nav_config: словарь с ключами 'speed_far_m_s', 'speed_near_m_s', 'speed_threshold_m'
        
    Выходы:
        (Vx_robot, Vy_robot): скорости в м/с
    """
    rx, ry = robot_data["center_m"]
    wx, wy = next_wp
    
    # 1. Расстояние до точки
    dist_to_wp = math.hypot(wx - rx, wy - ry)
    
    # 2. Выбор скорости
    V = nav_config["speed_far_m_s"] if dist_to_wp >= nav_config["speed_threshold_m"] else nav_config["speed_near_m_s"]
    
    # 3. Угол к цели в глобальной системе поля
    angle_to_wp = math.atan2(wy - ry, wx - rx)
    
    # 4. Скорости в глобальной системе поля
    Vx_field = V * math.cos(angle_to_wp)
    Vy_field = V * math.sin(angle_to_wp)
    
    # 5. Ориентация робота (учет поворота маркера)
    pts_top = robot_data["pts_top"]
    marker_heading = math.atan2(pts_top[1][1] - pts_top[0][1], pts_top[1][0] - pts_top[0][0])
    psi = marker_heading - math.pi / 2  # По условию: Ось X робота совпадает с Осью Y маркера
    
    # 6. Поворот вектора скорости из поля в локальную систему робота
    cos_psi = math.cos(psi)
    sin_psi = math.sin(psi)
    
    Vx_robot = Vx_field * cos_psi + Vy_field * sin_psi
    Vy_robot = -Vx_field * sin_psi + Vy_field * cos_psi
    
    return Vx_robot, Vy_robot

def stop_robot(ip: str) -> bool:
    """
    Безопасно останавливает робот (отправляет нулевые скорости).
    """
    return send_velocity(0.0, 0.0, 0.0, ip)

if __name__ == "__main__":
    # Тестовый запуск модуля (не выполняется при импорте)
    print("Запуск теста модуля robot_control...")
    cfg = load_robot_config()
    robot_ip = cfg.get("robot", {}).get("robot_id", "192.168.0.1")
    robot_port = cfg.get("robot", {}).get("robot_port", 80)
    
    print(f"Проверка подключения к {robot_ip}:{robot_port}...")
    sock = connect_to_robotino(robot_ip, robot_port)
    if sock:
        print("Проверка отправки скорости...")
        success = send_velocity(0.0, 0.0, 0.0, robot_ip)
        print(f"Результат: {'Успех' if success else 'Ошибка'}")
        sock.close()
    else:
        print("Подключение не установлено. Проверьте сеть и адрес робота.")