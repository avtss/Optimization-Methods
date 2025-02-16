import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt

def solve_quadratic_programming(x0):
    def objective(x):
        x1, x2 = x
        return 2*x1**2 + 3*x2**2 + 4*x1*x2 - 6*x1 - 3*x2

    def compute_lagrange_multipliers(x):
        x1, x2 = x
        lambda1 = max(0, 6 - 4*x1 - 2*x2)
        lambda2 = max(0, 3 - 4*x2 - 2*x1)
        return lambda1, lambda2

    max_iter = 100
    tolerance = 1e-6
    alpha = 0.1  # шаг спуска
    x = np.array(x0, dtype=float)
    history = []
    
    for _ in range(max_iter):
        grad = np.array([4*x[0] + 2*x[1] - 6, 6*x[1] + 4*x[0] - 3])
        lambda1, lambda2 = compute_lagrange_multipliers(x)
        step = alpha * (grad - np.array([lambda1, lambda2]))
        x_next = x - step
        
        x_next[0] = max(0, x_next[0])
        x_next[1] = max(0, x_next[1])
        
        if np.linalg.norm(x_next - x) < tolerance:
            break
        
        x = x_next
        history.append(x.copy())
    
    return x, history

def run_gui():
    def start_optimization():
        try:
            x0 = [float(entry_x0.get()), float(entry_x1.get())]
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения")
            return
        
        x_opt, history = solve_quadratic_programming(x0)
        result_label.config(text=f"Оптимальное решение: x1={x_opt[0]:.4f}, x2={x_opt[1]:.4f}")
        plot_results(history)

    def plot_results(history):
        history = np.array(history)
        plt.figure(figsize=(6,6))
        plt.plot(history[:,0], history[:,1], 'o-', label='Путь оптимизации')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Градиентный спуск с условиями Куна-Таккера')
        plt.legend()
        plt.show()

    root = tk.Tk()
    root.title("Квадратичное программирование")

    tk.Label(root, text="Начальная точка (x0, x1):").grid(row=0, column=0)
    entry_x0 = tk.Entry(root, width=10)
    entry_x0.insert(0, "0.5")
    entry_x0.grid(row=0, column=1)
    entry_x1 = tk.Entry(root, width=10)
    entry_x1.insert(0, "0.5")
    entry_x1.grid(row=0, column=2)

    start_button = tk.Button(root, text="Запуск", command=start_optimization)
    start_button.grid(row=1, column=0, columnspan=2)

    result_label = tk.Label(root, text="")
    result_label.grid(row=2, column=0, columnspan=3)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
