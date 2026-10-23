"""Численные методы для фиксированных модельных задач (без SciPy)."""

import math

import numpy as np

MAX_GRID_ELEMENTS = 2_000_000
HEAT_METHODS = ("Явная", "Неявная", "Кранка—Николсона")


def _positive(**parameters):
    for name, value in parameters.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}: требуется конечное положительное число.")


def _intervals(length, step, name, minimum=1):
    """Проверить делимость до создания сетки, включая слишком малые шаги."""
    ratio = length / step
    if not math.isfinite(ratio) or ratio > MAX_GRID_ELEMENTS:
        raise ValueError("Сетка слишком большая. Увеличьте шаг.")
    count = round(ratio)
    if count < minimum:
        raise ValueError(f"{name}: требуется не менее {minimum} интервалов.")
    if not math.isclose(ratio, count, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"{name}: шаг должен делить длину интервала без остатка.")
    return count


def _time_grid(length, final_time, h, tau):
    _positive(L=length, T=final_time, h=h, tau=tau)
    nx = _intervals(length, h, "h", minimum=4)
    nt = _intervals(final_time, tau, "tau")
    if (nx + 1) * (nt + 1) > MAX_GRID_ELEMENTS:
        raise ValueError("Сетка слишком большая. Увеличьте h или tau.")
    return np.linspace(0, length, nx + 1), np.linspace(0, final_time, nt + 1)


def classify_pde(coef_a, coef_b, coef_c):
    """Для A*u_xx + B*u_xy + C*u_yy: Δ = B² − 4AC.

    Нормировка защищает определение знака от масштаба коэффициентов.
    Сам дискриминант возвращается в исходном масштабе.
    """
    coefficients = (coef_a, coef_b, coef_c)
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("Коэффициенты должны быть конечными числами.")
    scale = max(abs(value) for value in coefficients)
    if scale == 0:
        raise ValueError("Все коэффициенты второго порядка равны нулю.")
    a, b, c = (value / scale for value in coefficients)
    delta = b * b - 4 * a * c
    tolerance = 1e-12 * (b * b + abs(4 * a * c))
    if abs(delta) <= tolerance:
        return 0.0, "ПАРАБОЛИЧЕСКИЙ"
    kind = "ЭЛЛИПТИЧЕСКИЙ" if delta < 0 else "ГИПЕРБОЛИЧЕСКИЙ"
    discriminant = delta * scale * scale
    if not math.isfinite(discriminant) or discriminant == 0:
        raise ValueError("Масштаб дискриминанта вне диапазона float; масштабируйте коэффициенты.")
    return discriminant, kind


def exact_heat_solution(x, time, a):
    _positive(a=a)
    if not math.isfinite(time) or time < 0:
        raise ValueError("Время должно быть конечным неотрицательным числом.")
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Координаты должны быть конечными числами.")
    return np.exp(-4 * np.pi ** 2 * a * time) * np.sin(2 * np.pi * x)


def calculate_max_error(numerical_solution, exact_solution):
    numerical = np.asarray(numerical_solution, dtype=float)
    exact = np.asarray(exact_solution, dtype=float)
    if numerical.shape != exact.shape:
        raise ValueError("Решения должны иметь одинаковый размер.")
    if numerical.size == 0:
        raise ValueError("Массивы решений не должны быть пустыми.")
    if not np.all(np.isfinite(numerical)) or not np.all(np.isfinite(exact)):
        raise ValueError("Решения должны содержать только конечные числа.")
    with np.errstate(over="ignore"):
        error = float(np.max(np.abs(numerical - exact)))
    if not math.isfinite(error):
        raise ValueError("Ошибка вне диапазона float.")
    return error


def solve_tridiagonal(lower, diagonal, upper, rhs):
    """Метод прогонки: нижняя/верхняя диагонали длины N−1, остальные N.

    Прямой ход исключает нижнюю диагональ; обратный восстанавливает решение.
    В наших схемах матрица строго диагонально преобладает, перестановки не нужны.
    """
    lower, diagonal, upper, rhs = (
        np.asarray(values, dtype=float).copy()
        for values in (lower, diagonal, upper, rhs)
    )
    n = diagonal.size
    if (n == 0 or any(v.ndim != 1 for v in (lower, diagonal, upper, rhs))
            or lower.size != n - 1 or upper.size != n - 1 or rhs.size != n):
        raise ValueError("Неверные размеры трёхдиагональной системы.")
    if not all(np.all(np.isfinite(v)) for v in (lower, diagonal, upper, rhs)):
        raise ValueError("Система должна содержать конечные числа.")
    for i in range(1, n):
        if diagonal[i - 1] == 0:
            raise ValueError("Нулевой ведущий элемент в методе прогонки.")
        factor = lower[i - 1] / diagonal[i - 1]
        diagonal[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]
    if np.any(diagonal == 0):
        raise ValueError("Вырожденная система в методе прогонки.")
    solution = np.empty(n)
    solution[-1] = rhs[-1] / diagonal[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = (rhs[i] - upper[i] * solution[i + 1]) / diagonal[i]
    return solution


def solve_heat(a, final_time, h, tau, method="Явная"):
    """u_t=a*u_xx, u(0,t)=u(1,t)=0, u(x,0)=sin(2πx).

    θ=0: явная схема; θ=1: неявная; θ=1/2: Кранк—Николсон.
    Возвращает x, t, u[n,i], r. Переданные шаги не изменяются.
    """
    _positive(a=a)
    if method not in HEAT_METHODS:
        raise ValueError("Неизвестная схема теплопроводности.")
    x, t = _time_grid(1.0, final_time, h, tau)
    r = (a * (tau / h)) / h
    if not math.isfinite(r) or r <= 0 or r > 1e12:
        raise ValueError("r вне допустимого численного диапазона (0, 10¹²].")
    if method == "Явная" and r > 0.5 + 1e-14:
        raise ValueError(f"Нарушена устойчивость: r = {r:g} > 0.5.")
    u = np.zeros((len(t), len(x)))
    u[0] = np.sin(2 * np.pi * x)
    u[:, 0] = u[:, -1] = 0
    theta = {"Явная": 0.0, "Неявная": 1.0, "Кранка—Николсона": 0.5}[method]
    size = len(x) - 2
    lower = np.full(size - 1, -theta * r)
    diagonal = np.full(size, 1 + 2 * theta * r)
    upper = lower.copy()
    for n in range(len(t) - 1):
        rhs = u[n, 1:-1] + (1 - theta) * r * (
            u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2]
        )
        if theta == 0:
            u[n + 1, 1:-1] = rhs
        else:
            u[n + 1, 1:-1] = solve_tridiagonal(lower, diagonal, upper, rhs)
    return x, t, u, r


def solve_heat_explicit(a, final_time, h, tau):
    return solve_heat(a, final_time, h, tau, "Явная")


def solve_heat_implicit(a, final_time, h, tau):
    return solve_heat(a, final_time, h, tau, "Неявная")


def solve_heat_crank_nicolson(a, final_time, h, tau):
    return solve_heat(a, final_time, h, tau, "Кранка—Николсона")


def heat_convergence(a, final_time, h, tau, method="Явная", levels=3):
    """Сгущаем h вдвое, tau вчетверо: r постоянен. Ошибка при t=T."""
    if not isinstance(levels, int) or not 1 <= levels <= 5:
        raise ValueError("Число уровней должно быть от 1 до 5.")
    rows = []
    for level in range(levels):
        step_x, step_t = h / 2 ** level, tau / 4 ** level
        x, t, u, r = solve_heat(a, final_time, step_x, step_t, method)
        error = calculate_max_error(u[-1], exact_heat_solution(x, t[-1], a))
        rows.append((step_x, step_t, r, error))
    return rows


def exact_wave_solution(x, time, c, length=1.0, amplitude=1.0):
    _positive(c=c, L=length)
    if not math.isfinite(time) or time < 0 or not math.isfinite(amplitude):
        raise ValueError("Время должно быть ≥ 0, амплитуда — конечным числом.")
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Координаты должны быть конечными числами.")
    return amplitude * np.sin(np.pi * x / length) * np.cos(np.pi * c * time / length)


def solve_wave(c, final_time, h, tau, length=1.0, amplitude=1.0):
    """Закреплённая струна, начальная скорость 0; возвращает x,t,u,CFL."""
    _positive(c=c)
    if not math.isfinite(amplitude):
        raise ValueError("Амплитуда должна быть конечным числом.")
    x, t = _time_grid(length, final_time, h, tau)
    courant = c * (tau / h)
    if not math.isfinite(courant) or courant <= 0 or courant > 1 + 1e-14:
        raise ValueError("Условие устойчивости: 0 < c·tau/h ≤ 1.")
    q = courant ** 2
    u = np.zeros((len(t), len(x)))
    u[0] = amplitude * np.sin(np.pi * x / length)
    u[:, 0] = u[:, -1] = 0
    # Разложение Тейлора: u¹ = u⁰ + tau*v⁰ + tau²*c²*u_xx⁰/2.
    u[1, 1:-1] = u[0, 1:-1] + q / 2 * (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2])
    for n in range(1, len(t) - 1):
        u[n + 1, 1:-1] = (2 * u[n, 1:-1] - u[n - 1, 1:-1]
                             + q * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2]))
    if not np.all(np.isfinite(u)):
        raise ValueError("Переполнение решения. Уменьшите амплитуду.")
    return x, t, u, courant


def solve_laplace(h=0.05, left=0.0, right=0.0, bottom=0.0, top=100.0,
                  epsilon=1e-6, max_iterations=10000):
    """Лаплас на единичном квадрате, равные шаги, метод Гаусса—Зейделя.

    u[j,i] соответствует (x_i,y_j). В углах — среднее смежных температур.
    Остановка по максимуму |u_xx+u_yy| в узлах, не по изменению итерации.
    Возвращает x,y,u,число итераций,невязку; незавершённый расчёт — ошибка.
    """
    _positive(h=h, epsilon=epsilon)
    if not all(math.isfinite(v) and abs(v) <= 1e12 for v in (left, right, bottom, top)):
        raise ValueError("Температуры должны быть конечными, по модулю ≤ 10¹².")
    if not isinstance(max_iterations, int) or not 1 <= max_iterations <= 100000:
        raise ValueError("Предел итераций должен быть целым от 1 до 100000.")
    intervals = _intervals(1.0, h, "h", minimum=2)
    points = intervals + 1
    # Ограничение работы синхронного GUI: не более 20 млн обновлений узлов.
    if points ** 2 > MAX_GRID_ELEMENTS or (points - 2) ** 2 * max_iterations > 20_000_000:
        raise ValueError("Слишком большой расчёт: увеличьте h или уменьшите предел итераций.")
    x = np.linspace(0, 1, points)
    u = np.full((points, points), (left + right + bottom + top) / 4)
    u[:, 0], u[:, -1] = left, right
    u[0, :], u[-1, :] = bottom, top
    u[0, 0], u[0, -1] = (bottom + left) / 2, (bottom + right) / 2
    u[-1, 0], u[-1, -1] = (top + left) / 2, (top + right) / 2
    for iteration in range(1, max_iterations + 1):
        for j in range(1, points - 1):
            for i in range(1, points - 1):
                u[j, i] = (u[j, i - 1] + u[j, i + 1] + u[j - 1, i] + u[j + 1, i]) / 4
        residual = float(np.max(np.abs(
            u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1]
            - 4 * u[1:-1, 1:-1]
        ))) / h ** 2
        if residual <= epsilon:
            return x, x.copy(), u, iteration, residual
    raise ValueError(f"Нет сходимости за {max_iterations} итераций; невязка {residual:.3g}. "
                     "Увеличьте предел итераций или epsilon.")
