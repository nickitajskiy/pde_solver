import numpy as np

length = 1.0
T = 0.2
h = 0.25
tau = 0.1

number_x_intervals = int(round(length / h))
number_t_intervals = int(round(T / tau))

number_x_points = number_x_intervals + 1
number_t_points = number_t_intervals + 1

x = np.linspace(0.0, length, number_x_points)
t = np.linspace(0.0, T, number_t_points)

u = np.zeros((number_t_points, number_x_points))

u[0, :] = np.sin(2 * np.pi * x)

u[:, 0] = 0.0
u[:, -1] = 0.0

np.set_printoptions(precision=3, suppress=True)

print("Координаты x:")
print(x)

print("\nМоменты времени t:")
print(t)

print("\nРазмер массива u:")
print(u.shape)

print("\nНачальное состояние массива u:")
print(u)