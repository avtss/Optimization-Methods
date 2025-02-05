import numpy as np

def gradient_descent(f, grad_f, x0, epsilon1, epsilon2, M, t):
    # Шаг 1: Задать x0, параметры epsilon1, epsilon2, M, шаг t
    xk = np.array(x0, dtype=float)
    k = 0
    
    print(f"{'k':<5}{'xk[0]':<12}{'xk[1]':<12}{'f(xk)':<12}{'grad f(xk)[0]':<15}{'grad f(xk)[1]':<15}{'||grad f(xk)||':<15}{'tk':<10}{'xk+1[0]':<12}{'xk+1[1]':<12}")
    
    # Шаг 2: Положить k = 0
    while True:
        # Шаг 3: Вычислить f(xk)
        fxk = f(xk)
        grad_fxk = grad_f(xk)
        norm_grad = np.linalg.norm(grad_fxk)
        
        # Шаг 4: Проверить выполнение критерия окончания | f(x*)| < ε1
        if norm_grad < epsilon1:
            return xk
        
        # Шаг 5: Проверить выполнение условия k ≥ M
        if k >= M:
            return xk
        
        # Шаг 6: Задать величину шага tk (у нас он фиксированный)
        
        # Шаг 7: Вычислить xk+1 = xk - tk * grad_f(xk)
        xk1 = xk - t * grad_fxk
        
        # Шаг 8: Проверить выполнение условия убывания функции
        if not (f(xk1) - fxk < 0 or abs(f(xk1) - fxk) < epsilon1 * np.linalg.norm(grad_fxk)):
            continue  # Если условие не выполняется, повторяем шаг 7
        
        # Вывод текущего шага
        print(f"{k:<5}{xk[0]:<12.6f}{xk[1]:<12.6f}{fxk:<12.6f}{grad_fxk[0]:<15.6f}{grad_fxk[1]:<15.6f}{norm_grad:<15.6f}{t:<10.6f}{xk1[0]:<12.6f}{xk1[1]:<12.6f}")
        
        # Шаг 9: Проверка сходимости
        if np.linalg.norm(xk1 - xk) < epsilon2 and abs(f(xk1) - fxk) < epsilon2:
            return xk1
        
        xk = xk1
        k += 1

# Пример использования
def f(x):
    return 2*x[0]**2 + x[0]*x[1] + x[1]**2  # Заданная функция

def grad_f(x):
    return np.array([4*x[0] + x[1], x[0] + 2*x[1]])  # Градиент функции

x0 = [0.5, 1]  # Начальная точка
epsilon1 = 0.1
epsilon2 = 0.15
M = 10
t = 0.24  # Фиксированный шаг

x_opt = gradient_descent(f, grad_f, x0, epsilon1, epsilon2, M, t)
print("Локальный минимум найден в точке:", x_opt)
print("Значение функции в этой точке:", f(x_opt))
