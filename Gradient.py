import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Символьные переменные
x1, x2 = sp.symbols('x1 x2')

# Функция для минимизации (по умолчанию)
default_function = 2*x1**2 + x1*x2 + x2**2

# Функция для вычисления градиента автоматически
def compute_gradient(func):
    grad = [sp.diff(func, var) for var in (x1, x2)]
    return sp.lambdify((x1, x2), grad, 'numpy')

# Градиент целевой функции
grad_f = compute_gradient(default_function)
f_func = sp.lambdify((x1, x2), default_function, 'numpy')

# Метод градиентного спуска с историей итераций
def gradient_descent(x0, step_size, epsilon1, epsilon2, max_iter):
    xk = np.array(x0, dtype=float)
    history = []
    
    for k in range(max_iter):
        fxk = f_func(xk[0], xk[1])
        grad_fxk = np.array(grad_f(xk[0], xk[1]))
        norm_grad = np.linalg.norm(grad_fxk)
        
        if norm_grad < epsilon1:
            break
        
        x_next = xk - step_size * grad_fxk
        
        history.append((k, xk[0], xk[1], fxk, grad_fxk[0], grad_fxk[1], norm_grad, step_size, x_next[0], x_next[1]))
        
        if np.linalg.norm(x_next - xk) < epsilon2 and abs(f_func(x_next[0], x_next[1]) - fxk) < epsilon2:
            break
        
        xk = x_next
    
    return xk, history

# Интерфейс приложения
def run_gui():
    def start_optimization():
        try:
            x0 = [float(entry_x0.get()), float(entry_y0.get())]
            step_size = float(entry_step.get())
            epsilon1 = float(entry_eps1.get())
            epsilon2 = float(entry_eps2.get())
            max_iter = int(entry_max_iter.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения")
            return
        
        x_min, history = gradient_descent(x0, step_size, epsilon1, epsilon2, max_iter)
        result_label.config(text=f"Найденный минимум: ({x_min[0]:.4f}, {x_min[1]:.4f})")
        update_table(history)
        plot_results(history)
    
    def update_table(history):
        for row in table.get_children():
            table.delete(row)
        
        for data in history:
            table.insert("", "end", values=[round(val, 4) if isinstance(val, float) else val for val in data])
    
    def plot_results(history):
        history = np.array(history)
        
        plt.figure(figsize=(6,6))
        plt.plot(history[:,1], history[:,2], 'o-')
        
        x_vals = np.linspace(-2, 2, 100)
        y_vals = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f_func(X, Y)
        plt.contour(X, Y, Z, levels=30, cmap='viridis')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Градиентный спуск')
        plt.show()
    
    root = tk.Tk()
    root.title("Градиентный спуск")
    
    tk.Label(root, text="Начальная точка (x0, y0):").grid(row=0, column=0)
    entry_x0 = tk.Entry(root, width=5)
    entry_x0.insert(0, "0.5")
    entry_x0.grid(row=0, column=1)
    entry_y0 = tk.Entry(root, width=5)
    entry_y0.insert(0, "1.0")
    entry_y0.grid(row=0, column=2)
    
    tk.Label(root, text="Шаг спуска:").grid(row=1, column=0)
    entry_step = tk.Entry(root, width=10)
    entry_step.insert(0, "0.24")
    entry_step.grid(row=1, column=1)
    
    tk.Label(root, text="Epsilon 1:").grid(row=2, column=0)
    entry_eps1 = tk.Entry(root, width=10)
    entry_eps1.insert(0, "0.1")
    entry_eps1.grid(row=2, column=1)
    
    tk.Label(root, text="Epsilon 2:").grid(row=3, column=0)
    entry_eps2 = tk.Entry(root, width=10)
    entry_eps2.insert(0, "0.15")
    entry_eps2.grid(row=3, column=1)
    
    tk.Label(root, text="Максимальное число итераций:").grid(row=4, column=0)
    entry_max_iter = tk.Entry(root, width=10)
    entry_max_iter.insert(0, "10")
    entry_max_iter.grid(row=4, column=1)
    
    start_button = tk.Button(root, text="Запуск", command=start_optimization)
    start_button.grid(row=5, column=0, columnspan=2)
    
    result_label = tk.Label(root, text="")
    result_label.grid(row=6, column=0, columnspan=3)
    
    columns = ("k", "xk[0]", "xk[1]", "f(xk)", "grad_x[0]", "grad_x[1]", "||grad||", "t", "xk+1[0]", "xk+1[1]")
    table = ttk.Treeview(root, columns=columns, show="headings")
    
    for col in columns:
        table.heading(col, text=col)
        table.column(col, width=80)
    
    table.grid(row=7, column=0, columnspan=3)
    
    root.mainloop()

if __name__ == "__main__":
    run_gui()
