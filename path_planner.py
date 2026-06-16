import numpy as np
import heapq
import math
import random
from scipy.interpolate import splprep, splev
from collections import deque

s_spline = 0.05

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================
def _find_valid_start_cell(grid, start, rows, cols, max_radius=3):
    """
    Ищет ближайшую свободную ячейку (0) в заданном радиусе от стартовой позиции.
    Это предотвращает сбой планировщика, если робот физически находится в свободном месте,
    но из-за safety_margin его стартовая ячейка в сетке помечена как препятствие (1).
    
    Входы:
        grid: бинарная сетка препятствий
        start: кортеж (x, y) текущей стартовой ячейки
        rows, cols: размеры сетки
        max_radius: максимальный радиус поиска в клетках (3 клетки = 15 см при cell_size=0.05)
    
    Выходы:
        Кортеж (x, y) новой безопасной стартовой ячейки, или None, если выход не найден.
    """
    start_x, start_y = start
    
    # Если стартовая ячейка уже свободна и в пределах поля, возвращаем её
    if 0 <= start_y < rows and 0 <= start_x < cols and grid[start_y, start_x] == 0:
        return start

    # BFS (поиск в ширину) для гарантированного нахождения ближайшего выхода
    queue = deque([(start_x, start_y, 0)])  # (x, y, расстояние)
    visited = set()
    visited.add((start_x, start_y))

    while queue:
        cx, cy, dist = queue.popleft()
        
        # Если превысили максимальный радиус поиска, прекращаем ветку
        if dist > max_radius:
            continue
            
        # Если нашли свободную ячейку, возвращаем её как новый старт
        if 0 <= cy < rows and 0 <= cx < cols and grid[cy, cx] == 0:
            return (cx, cy)

        # Проверяем 4-х соседей (безопаснее для выхода из узких щелей, чем 8-связность)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited:
                if 0 <= nx < cols and 0 <= ny < rows:
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))
                    
    # Если ничего не найдено в радиусе, робот действительно заперт
    return None
    
def _create_grid(width_m, height_m, obstacles, cell_size, safety_margin, obs_radius_m=0.15):
    cols = int(width_m / cell_size)
    rows = int(height_m / cell_size)
    grid = np.zeros((rows, cols), dtype=np.int8)
    
    inflate_edges = int(math.ceil(safety_margin / cell_size))
    if inflate_edges > 0:
        grid[:inflate_edges, :] = 1
        grid[-inflate_edges:, :] = 1
        grid[:, :inflate_edges] = 1
        grid[:, -inflate_edges:] = 1

    inflate_obs = int(math.ceil((obs_radius_m + safety_margin) / cell_size))
    for obs in obstacles:
        cx, cy = obs["center_m"]
        c_col = int(cx / cell_size)
        c_row = int(cy / cell_size)
        r_min = max(0, c_row - inflate_obs)
        r_max = min(rows, c_row + inflate_obs + 1)
        c_min = max(0, c_col - inflate_obs)
        c_max = min(cols, c_col + inflate_obs + 1)
        grid[r_min:r_max, c_min:c_max] = 1

    return grid, rows, cols

def _heuristic(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def _get_neighbors(node):
    return [
        (node[0]-1, node[1]-1), (node[0], node[1]-1), (node[0]+1, node[1]-1),
        (node[0]-1, node[1]),                      (node[0]+1, node[1]),
        (node[0]-1, node[1]+1), (node[0], node[1]+1), (node[0]+1, node[1]+1)
    ]

def astar(grid, start, goal):
    rows, cols = grid.shape
    if not (0 <= start[0] < cols and 0 <= start[1] < rows) or grid[start[1], start[0]] == 1:
        return None
    if not (0 <= goal[0] < cols and 0 <= goal[1] < rows) or grid[goal[1], goal[0]] == 1:
        return None

    open_set = []
    counter = 0
    heapq.heappush(open_set, (0, counter, start))
    came_from = {}
    g_score = {start: 0}
    open_set_hash = {start}

    while open_set:
        _, _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in _get_neighbors(current):
            nx, ny = neighbor
            if not (0 <= nx < cols and 0 <= ny < rows) or grid[ny, nx] == 1:
                continue

            move_cost = 1.0 if abs(nx - current[0]) + abs(ny - current[1]) == 1 else 1.414
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + _heuristic(neighbor, goal)
                if neighbor not in open_set_hash:
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
                    open_set_hash.add(neighbor)
    return None

def _check_collision(x, y, obstacles, robot_radius):
    for obs in obstacles:
        dist = math.hypot(x - obs["center_m"][0], y - obs["center_m"][1])
        if dist < (robot_radius+0.02):
            return True
    return False

def _check_line_collision(p1, p2, obstacles, robot_radius, num_checks=10):
    for i in range(num_checks + 1):
        t = i / num_checks
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        if _check_collision(x, y, obstacles, robot_radius):
            return True
    return False

# ================= АЛГОРИТМЫ ПЛАНИРОВАНИЯ =================

def plan_path_rrt_star(start, goal, width_m, height_m, obstacles, robot_radius, max_iter=1500, step_size=0.1, search_radius=0.3):
    if _check_collision(start[0], start[1], obstacles, robot_radius) or _check_collision(goal[0], goal[1], obstacles, robot_radius):
        return None

    tree = [{'x': start[0], 'y': start[1], 'parent': None, 'cost': 0.0}]
    
    for _ in range(max_iter):
        if random.random() < 0.05:
            rand_pt = goal
        else:
            rand_pt = (random.uniform(0, width_m), random.uniform(0, height_m))

        nearest_idx = min(range(len(tree)), key=lambda i: math.hypot(tree[i]['x'] - rand_pt[0], tree[i]['y'] - rand_pt[1]))
        nearest_node = tree[nearest_idx]

        dist = math.hypot(rand_pt[0] - nearest_node['x'], rand_pt[1] - nearest_node['y'])
        if dist > step_size:
            new_x = nearest_node['x'] + (rand_pt[0] - nearest_node['x']) / dist * step_size
            new_y = nearest_node['y'] + (rand_pt[1] - nearest_node['y']) / dist * step_size
        else:
            new_x, new_y = rand_pt

        if _check_line_collision((nearest_node['x'], nearest_node['y']), (new_x, new_y), obstacles, robot_radius):
            continue

        min_cost = nearest_node['cost'] + math.hypot(new_x - nearest_node['x'], new_y - nearest_node['y'])
        best_parent_idx = nearest_idx

        for i, node in enumerate(tree):
            if math.hypot(node['x'] - new_x, node['y'] - new_y) < search_radius:
                if not _check_line_collision((node['x'], node['y']), (new_x, new_y), obstacles, robot_radius):
                    new_cost = node['cost'] + math.hypot(new_x - node['x'], new_y - node['y'])
                    if new_cost < min_cost:
                        min_cost = new_cost
                        best_parent_idx = i

        new_idx = len(tree)
        tree.append({'x': new_x, 'y': new_y, 'parent': best_parent_idx, 'cost': min_cost})

        for i, node in enumerate(tree):
            if i == new_idx: continue
            dist_to_new = math.hypot(node['x'] - new_x, node['y'] - new_y)
            if dist_to_new < search_radius:
                new_cost_via_new = min_cost + dist_to_new
                if new_cost_via_new < node['cost'] and not _check_line_collision((new_x, new_y), (node['x'], node['y']), obstacles, robot_radius):
                    tree[i]['parent'] = new_idx
                    tree[i]['cost'] = new_cost_via_new

        if math.hypot(new_x - goal[0], new_y - goal[1]) < step_size:
            path = [(goal[0], goal[1])]
            curr = new_idx
            while curr is not None:
                path.append((tree[curr]['x'], tree[curr]['y']))
                curr = tree[curr]['parent']
            path.append(start)
            return path[::-1]

    return None

def plan_path_apf(start, goal, width_m, height_m, obstacles, robot_radius, att_gain=1.0, rep_gain=3.0, influence_radius=0.4, step_size=0.05, max_iter=500, goal_thresh=0.1):
    if _check_collision(start[0], start[1], obstacles, robot_radius):
        return None

    path = [start]
    current = list(start)

    for _ in range(max_iter):
        dist_to_goal = math.hypot(goal[0] - current[0], goal[1] - current[1])
        if dist_to_goal < goal_thresh:
            path.append(goal)
            return path

        F_att_x = att_gain * (goal[0] - current[0])
        F_att_y = att_gain * (goal[1] - current[1])

        F_rep_x, F_rep_y = 0.0, 0.0
        for obs in obstacles:
            dist = math.hypot(obs["center_m"][0] - current[0], obs["center_m"][1] - current[1])
            safe_dist = influence_radius + robot_radius
            if dist < safe_dist and dist > 0.01:
                rep_mag = rep_gain * (1.0 / dist - 1.0 / safe_dist) * (1.0 / (dist ** 2))
                F_rep_x += rep_mag * (current[0] - obs["center_m"][0]) / dist
                F_rep_y += rep_mag * (current[1] - obs["center_m"][1]) / dist

        F_x = F_att_x + F_rep_x
        F_y = F_att_y + F_rep_y

        if math.hypot(F_x, F_y) < 0.01:
            F_x += random.uniform(-0.5, 0.5)
            F_y += random.uniform(-0.5, 0.5)

        force_mag = math.hypot(F_x, F_y)
        current[0] += (F_x / force_mag) * step_size
        current[1] += (F_y / force_mag) * step_size

        current[0] = max(0.0, min(width_m, current[0]))
        current[1] = max(0.0, min(height_m, current[1]))

        path.append(tuple(current))

    return path

def smooth_path_spline(path, num_points=50):
    """
    Сглаживает ломаный путь с помощью параметрического кубического сплайна.
    Включает защиту от дубликатов точек (частая проблема RRT*).
    """
    if len(path) < 3:
        return path
    
    x = [p[0] for p in path]
    y = [p[1] for p in path]
    
    # 1. ОЧИСТКА: Удаляем точки, которые находятся ближе 1 мм друг к другу
    clean_x = [x[0]]
    clean_y = [y[0]]
    
    for i in range(1, len(x)):
        dist = math.hypot(x[i] - clean_x[-1], y[i] - clean_y[-1])
        if dist > 1e-3:  # 1e-3 метра = 1 мм
            clean_x.append(x[i])
            clean_y.append(y[i])
            
    # Если после очистки осталось меньше 3 уникальных точек, сплайн строить нельзя
    if len(clean_x) < 3:
        return path  # Возвращаем исходный путь, чтобы не ломать логику программы
    
    # 2. ПОСТРОЕНИЕ СПЛАЙНА с защитой от сбоев
    k = min(3, len(clean_x) - 1)  # Степень сплайна (максимум 3)
    
    try:
        # Пробуем построить кубический сплайн (k=3)
        tck, u = splprep([clean_x, clean_y], s = s_spline, k=k)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return list(zip(x_new.tolist(), y_new.tolist()))
        
    except ValueError:
        # Если кубический сплайн не смог (например, все точки на одной прямой), 
        # пробуем линейную интерполяцию (k=1), которая работает всегда для >= 2 точек
        try:
            tck, u = splprep([clean_x, clean_y], s = s_spline, k=1)
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)
            return list(zip(x_new.tolist(), y_new.tolist()))
        except Exception:
            # Полный откат: если scipy всё равно отказывается, возвращаем очищенный путь как есть
            return list(zip(clean_x, clean_y))

def bidirectional_dijkstra(grid, start, goal):
    """
    Двунаправленный поиск Дейкстры.
    Возвращает кратчайший путь или None, если путь невозможен.
    """
    rows, cols = grid.shape
    if not (0 <= start[0] < cols and 0 <= start[1] < rows) or grid[start[1], start[0]] == 1:
        return None
    if not (0 <= goal[0] < cols and 0 <= goal[1] < rows) or grid[goal[1], goal[0]] == 1:
        return None

    # Инициализация для поиска от старта
    open_start = []
    heapq.heappush(open_start, (0, start))
    g_start = {start: 0}
    came_from_start = {}

    # Инициализация для поиска от цели
    open_goal = []
    heapq.heappush(open_goal, (0, goal))
    g_goal = {goal: 0}
    came_from_goal = {}

    meet_node = None
    best_path_length = float('inf')

    while open_start and open_goal:
        # Всегда расширяем тот фронт, который сейчас "дешевле" (меньше элементов или меньше g_score)
        # Для простоты и стабильности будем чередовать или брать меньший по размеру
        if len(open_start) <= len(open_goal):
            current_d, current = heapq.heappop(open_start)
            current_g_dict = g_start
            current_came_from = came_from_start
            other_g_dict = g_goal
            direction = 'start'
        else:
            current_d, current = heapq.heappop(open_goal)
            current_g_dict = g_goal
            current_came_from = came_from_goal
            other_g_dict = g_start
            direction = 'goal'

        # Если текущая стоимость уже больше лучшего найденного пути, можно останавливаться
        if current_d > best_path_length:
            break

        for neighbor in _get_neighbors(current):
            nx, ny = neighbor
            if not (0 <= nx < cols and 0 <= ny < rows) or grid[ny, nx] == 1:
                continue

            move_cost = 1.0 if abs(nx - current[0]) + abs(ny - current[1]) == 1 else 1.414
            tentative_g = current_g_dict[current] + move_cost

            if tentative_g < current_g_dict.get(neighbor, float('inf')):
                current_came_from[neighbor] = current
                current_g_dict[neighbor] = tentative_g
                
                # Если сосед уже посещен в противоположном направлении, мы нашли пересечение!
                if neighbor in other_g_dict:
                    total_len = tentative_g + other_g_dict[neighbor]
                    if total_len < best_path_length:
                        best_path_length = total_len
                        meet_node = neighbor

                # Добавляем в очередь, если его там нет (упрощенная проверка через g_score)
                # В полноценной реализации нужен open_set_hash, но для малой сетки heapq справится
                heapq.heappush(open_start if direction == 'start' else open_goal, (tentative_g, neighbor))

    # Если пересечение не найдено
    if meet_node is None:
        return None

    # Восстановление пути от meet_node к старту
    path_start = []
    curr = meet_node
    while curr in came_from_start:
        path_start.append(curr)
        curr = came_from_start[curr]
    path_start.append(start)
    path_start.reverse()

    # Восстановление пути от meet_node к цели
    path_goal = []
    curr = meet_node
    while curr in came_from_goal:
        curr = came_from_goal[curr]
        path_goal.append(curr)
    # Не добавляем meet_node второй раз, поэтому начинаем с следующего узла
    # Но так как цикл выше идет от meet_node к goal, нам нужно его развернуть
    # Исправленная логика восстановления для goal:
    path_goal_reversed = []
    curr = meet_node
    while curr in came_from_goal:
        path_goal_reversed.append(curr)
        curr = came_from_goal[curr]
    path_goal_reversed.append(goal)
    
    # Объединяем: путь от старта до meet_node (уже развернут) + путь от meet_node к цели (без дублирования meet_node)
    final_path = path_start + path_goal_reversed[1:]
    
    return final_path

def greedy_best_first(grid, start, goal):
    """
    Жадный алгоритм поиска по первому наилучшему совпадению.
    Выбирает следующую ячейку исключительно на основе эвристики (расстояния до цели),
    игнорируя стоимость уже пройденного пути.
    """
    rows, cols = grid.shape
    if not (0 <= start[0] < cols and 0 <= start[1] < rows) or grid[start[1], start[0]] == 1:
        return None
    if not (0 <= goal[0] < cols and 0 <= goal[1] < rows) or grid[goal[1], goal[0]] == 1:
        return None

    open_set = []
    counter = 0  # Счетчик для разрешения коллизий в heapq при одинаковой эвристике
    heapq.heappush(open_set, (_heuristic(start, goal), counter, start))
    
    came_from = {}
    visited = set()
    visited.add(start)

    while open_set:
        _, _, current = heapq.heappop(open_set)

        # Если достигли цели, восстанавливаем путь
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        # Проверяем всех соседей
        for neighbor in _get_neighbors(current):
            nx, ny = neighbor
            # Если сосед в пределах поля и не является препятствием
            if 0 <= nx < cols and 0 <= ny < rows and grid[ny, nx] == 0:
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    counter += 1
                    # В приоритетную очередь кладем ТОЛЬКО эвристику (жадный выбор)
                    heapq.heappush(open_set, (_heuristic(neighbor, goal), counter, neighbor))
    
    # Если очередь опустела, а цель не достигнута
    return None

# ================= ЕДИНАЯ ТОЧКА ВХОДА =================

def plan_path(robot_m, target_m, width_m, height_m, obstacles, planner_config):
    """
    Универсальная функция планирования.
    planner_config: словарь с ключами 'type', 'cell_size', 'safety_margin', 'obs_radius_m' и специфичными параметрами.
    """
    if robot_m is None or target_m is None:
        return None
        
    planner_type = planner_config.get("type", "astar").lower()
    base_radius = planner_config.get("obs_radius_m", 0.225)
    safety_margin = planner_config.get("safety_margin", 0.05)
    effective_radius = base_radius + safety_margin 

    # Общая логика для всех алгоритмов, работающих по сетке
    if planner_type in ["astar", "bi_dijkstra", "greedy"]:
        cell_size = planner_config.get("cell_size", 0.05)
        grid, rows, cols = _create_grid(width_m, height_m, obstacles, cell_size, safety_margin, base_radius)
        
        start_grid = (int(robot_m[0] / cell_size), int(robot_m[1] / cell_size))
        goal_grid = (int(target_m[0] / cell_size), int(target_m[1] / cell_size))
    
        valid_start = _find_valid_start_cell(grid, start_grid, rows, cols, max_radius=3)
        
        if valid_start is None:
            print("[ERROR] Робот физически заперт или инфляция препятствий слишком велика (нет выхода в радиусе 3 клеток)!")
            return None
            
        start = valid_start # Используем найденную безопасную ячейку как старт
        
        # Далее вызываем конкретный алгоритм
        if planner_type == "astar":
            path_grid = astar(grid, start, goal_grid)
        elif planner_type == "bi_dijkstra":
            path_grid = bidirectional_dijkstra(grid, start, goal_grid)
        elif planner_type == "greedy":
            path_grid = greedy_best_first(grid, start, goal_grid)
            
        return [(x * cell_size, y * cell_size) for x, y in path_grid] if path_grid else None

    elif planner_type == "rrt_star":
        rrt_cfg = planner_config.get("rrt_star", {})
        return plan_path_rrt_star(
            robot_m, target_m, width_m, height_m, obstacles, 
            effective_radius,
            max_iter=rrt_cfg.get("max_iterations", 1500),
            step_size=rrt_cfg.get("step_size", 0.15),
            search_radius=rrt_cfg.get("search_radius", 0.35)
        )

    elif planner_type == "apf":
        apf_cfg = planner_config.get("apf", {})
        return plan_path_apf(
            robot_m, target_m, width_m, height_m, obstacles, 
            effective_radius,
            att_gain=apf_cfg.get("attractive_gain", 1.0),
            rep_gain=apf_cfg.get("repulsive_gain", 3.0),
            influence_radius=apf_cfg.get("influence_radius", 0.4),
            step_size=apf_cfg.get("step_size", 0.05),
            max_iter=apf_cfg.get("max_iterations", 500),
            goal_thresh=apf_cfg.get("goal_threshold", 0.1)
        )
    else:
        print(f"[ERROR] Неизвестный тип планировщика: {planner_type}")
        return None

def generate_grid_for_debug(width_m, height_m, obstacles, cell_size, safety_margin, obs_radius_m=0.15):
    """Возвращает бинарную сетку для визуализации (только для A*)."""
    grid, _, _ = _create_grid(width_m, height_m, obstacles, cell_size, safety_margin, obs_radius_m)
    return grid

def calculate_tracking_metrics(actual_path, planned_path):
    """
    Вычисляет метрики качества следования траектории.
    Возвращает: (фактический путь в мм, планируемый путь в мм, MSE в мм², R²)
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