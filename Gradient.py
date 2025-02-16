import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from mpl_toolkits.mplot3d import Axes3D

# Функция и её градиент
def f(x):
    return 2 * x[0]**2 + x[0] * x[1] + x[1]**2  # Новая функция

def grad_f(x):
    return np.array([4*x[0] + x[1], x[0] + 2*x[1]])  # Градиент функции

# Алгоритм градиентного спуска
def gradient_descent(x0, epsilon, epsilon1, epsilon2, M):
    xk = np.array(x0, dtype=float)
    k = 0
    trajectory = [xk.copy()]  # Для визуализации траектории

    while k < M:
        fxk = f(xk)

        if abs(fxk) < epsilon1:  # Проверка критерия окончания
            break

        tk = 0.1  # Фиксированный шаг
        grad_fxk = grad_f(xk)
        xk_next = xk - tk * grad_fxk

        if f(xk_next) - fxk >= 0 and abs(f(xk_next) - fxk) >= epsilon * np.linalg.norm(grad_fxk):
            continue  # Повторяем с другим tk

        if np.linalg.norm(xk_next - xk) < epsilon2 and abs(f(xk_next) - fxk) < epsilon2:
            xk = xk_next
            break

        xk = xk_next
        trajectory.append(xk.copy())  # Запоминаем траекторию
        k += 1

    return xk, trajectory

# Функция для запуска градиентного спуска
def run_gradient_descent():
    x0 = np.array([float(entry_x0.get()), float(entry_y0.get())])
    epsilon = float(entry_epsilon.get())
    epsilon1 = float(entry_epsilon1.get())
    epsilon2 = float(entry_epsilon2.get())
    M = int(entry_M.get())

    x_star, trajectory = gradient_descent(x0, epsilon, epsilon1, epsilon2, M)
    label_result.config(text=f"Оптимальное решение: ({x_star[0]:.5f}, {x_star[1]:.5f})")

    # Отображение 3D-графика
    plot_gradient_descent_3D(trajectory)

# Построение 3D-графика градиентного спуска
def plot_gradient_descent_3D(trajectory):
    trajectory = np.array(trajectory)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 2 * X**2 + X * Y + Y**2  # Новая функция

    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)  # Поверхность функции
    ax.plot(trajectory[:, 0], trajectory[:, 1], f(trajectory.T), marker="o", color="red", linestyle="-", label="Траектория")

    ax.set_title("3D Градиентный спуск")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    ax.legend()

    # Отображение графика в Tkinter
    for widget in frame_graph.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame_graph)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Создание GUI
root = tk.Tk()
root.title("Градиентный спуск (3D)")

frame_controls = tk.Frame(root)
frame_controls.pack(side=tk.LEFT, padx=10, pady=10)

frame_graph = tk.Frame(root)
frame_graph.pack(side=tk.RIGHT, padx=10, pady=10)

# Поля для ввода параметров
tk.Label(frame_controls, text="Начальные координаты:").pack()
entry_x0 = tk.Entry(frame_controls)
entry_x0.pack()
entry_x0.insert(0, "10")  # По умолчанию x0 = 10

entry_y0 = tk.Entry(frame_controls)
entry_y0.pack()
entry_y0.insert(0, "-5")  # По умолчанию y0 = -5

tk.Label(frame_controls, text="epsilon:").pack()
entry_epsilon = tk.Entry(frame_controls)
entry_epsilon.pack()
entry_epsilon.insert(0, "1e-5")

tk.Label(frame_controls, text="epsilon1:").pack()
entry_epsilon1 = tk.Entry(frame_controls)
entry_epsilon1.pack()
entry_epsilon1.insert(0, "1e-6")

tk.Label(frame_controls, text="epsilon2:").pack()
entry_epsilon2 = tk.Entry(frame_controls)
entry_epsilon2.pack()
entry_epsilon2.insert(0, "1e-6")

tk.Label(frame_controls, text="Макс. итерации M:").pack()
entry_M = tk.Entry(frame_controls)
entry_M.pack()
entry_M.insert(0, "1000")

# Кнопка запуска
btn_run = tk.Button(frame_controls, text="Запустить", command=run_gradient_descent)
btn_run.pack(pady=5)

# Вывод результата
label_result = tk.Label(frame_controls, text="Оптимальное решение: ")
label_result.pack()

root.mainloop()
