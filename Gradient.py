import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

def gradient_descent(f, grad_f, x0, epsilon1, epsilon2, M, t):
    xk = np.array(x0, dtype=float)
    k = 0
    history = []
    
    while True:
        fxk = f(xk)
        grad_fxk = grad_f(xk)
        norm_grad = np.linalg.norm(grad_fxk)
        
        if norm_grad < epsilon1:
            return xk, history
        
        if k >= M:
            return xk, history
        
        xk1 = xk - t * grad_fxk
        
        if not (f(xk1) - fxk < 0 or abs(f(xk1) - fxk) < epsilon1 * np.linalg.norm(grad_fxk)):
            continue
        
        history.append((k, xk[0], xk[1], fxk, grad_fxk[0], grad_fxk[1], norm_grad, t, xk1[0], xk1[1]))
        
        if np.linalg.norm(xk1 - xk) < epsilon2 and abs(f(xk1) - fxk) < epsilon2:
            return xk1, history
        
        xk = xk1
        k += 1

def f(x):
    return 2*x[0]**2 + x[0]*x[1] + x[1]**2

def grad_f(x):
    return np.array([4*x[0] + x[1], x[0] + 2*x[1]])

def run_gradient_descent():
    x0 = [float(entry_x0_0.get()), float(entry_x0_1.get())]
    epsilon1 = float(entry_epsilon1.get())
    epsilon2 = float(entry_epsilon2.get())
    M = int(entry_M.get())
    t = float(entry_t.get())
    
    x_opt, history = gradient_descent(f, grad_f, x0, epsilon1, epsilon2, M, t)
    result_text.set(f"Локальный минимум: {x_opt}\nЗначение функции: {f(x_opt)}")
    
    for row in table.get_children():
        table.delete(row)
    
    for data in history:
        table.insert("", "end", values=data)
    
    history = np.array(history)
    plt.figure()
    plt.plot(history[:, 0], history[:, 3], marker='o', linestyle='-')
    plt.xlabel("Итерация")
    plt.ylabel("Значение функции")
    plt.title("Сходимость градиентного спуска")
    plt.grid()
    plt.show()

root = tk.Tk()
root.title("Градиентный спуск")

tk.Label(root, text="x0[0]:").grid(row=0, column=0)
entry_x0_0 = tk.Entry(root)
entry_x0_0.grid(row=0, column=1)

tk.Label(root, text="x0[1]:").grid(row=1, column=0)
entry_x0_1 = tk.Entry(root)
entry_x0_1.grid(row=1, column=1)

tk.Label(root, text="epsilon1:").grid(row=2, column=0)
entry_epsilon1 = tk.Entry(root)
entry_epsilon1.grid(row=2, column=1)

tk.Label(root, text="epsilon2:").grid(row=3, column=0)
entry_epsilon2 = tk.Entry(root)
entry_epsilon2.grid(row=3, column=1)

tk.Label(root, text="M:").grid(row=4, column=0)
entry_M = tk.Entry(root)
entry_M.grid(row=4, column=1)

tk.Label(root, text="t:").grid(row=5, column=0)
entry_t = tk.Entry(root)
entry_t.grid(row=5, column=1)

tk.Button(root, text="Запустить", command=run_gradient_descent).grid(row=6, column=0, columnspan=2)

result_text = tk.StringVar()
tk.Label(root, textvariable=result_text).grid(row=7, column=0, columnspan=2)

columns = ("k", "xk[0]", "xk[1]", "f(xk)", "grad f(xk)[0]", "grad f(xk)[1]", "||grad f(xk)||", "tk", "xk+1[0]", "xk+1[1]")
table = ttk.Treeview(root, columns=columns, show="headings")

for col in columns:
    table.heading(col, text=col)
    table.column(col, width=100)

table.grid(row=8, column=0, columnspan=2)

root.mainloop()
