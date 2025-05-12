import numpy as np

def optimize(func, max_iter, population_size, x_min, x_max, y_min, y_max, nb, nc, nd, mutation, tolerance_steps):
    """
    Функция оптимизации методом, похожим на генетический алгоритм
    
    Параметры:
    func - целевая функция для минимизации
    max_iter - максимальное количество итераций
    population_size - размер популяции
    x_min, x_max, y_min, y_max - границы поиска
    nb - количество лучших особей для отбора
    nc - коэффициент размножения лучших особей
    nd - количество потомков для сохранения
    mutation - начальный уровень мутации
    tolerance_steps - количество шагов без улучшения для остановки
    """
    
    # Размерность задачи (2D - x и y)
    dim = 2
    
    # Инициализация начальной популяции случайными значениями в заданных границах
    population = np.random.uniform(low=[x_min, y_min], 
                                 high=[x_max, y_max],
                                 size=(population_size, dim))
    
    # История оптимизации для сохранения результатов
    history = []
    
    # Лучшие значения
    best_fitness = np.inf  # Наилучшее значение функции
    best_position = None   # Позиция наилучшего значения

    # Расчет изменения мутации на каждой итерации
    delta_mutation = mutation/max_iter
    mutation_rate = mutation  # Текущий уровень мутации

    # Параметры для критерия остановки
    tolerance = 1e-16  # Минимальное значимое изменение
    no_improvement_steps = 0  # Счетчик шагов без улучшения
    
    # Основной цикл оптимизации
    for iteration in range(max_iter):
        # Вычисление значений функции для всех особей
        fitness = np.array([func(*ind) for ind in population])
        
        # Отбор nb лучших особей
        best_indices = np.argsort(fitness)[:nb]
        Sb = population[best_indices].copy()

        # Размножение лучших особей (каждую повторяем nc раз)
        Sm = np.repeat(Sb, nc, axis=0)
        
        # Применение мутации к потомкам
        mutation = mutation_rate * np.random.uniform(-0.5, 0.5, Sm.shape)
        Sm = Sm + mutation

        # Ограничение значений в допустимых границах
        Sm[:, 0] = np.clip(Sm[:, 0], x_min, x_max)
        Sm[:, 1] = np.clip(Sm[:, 1], y_min, y_max)

        # Оценка качества потомков
        Sm_fitness = np.array([func(*ind) for ind in Sm])
        
        # Отбор nd лучших потомков
        best_Sm_indices = np.argsort(Sm_fitness)[:nd]
        Sm = Sm[best_Sm_indices]
        
        # Объединение исходной популяции и потомков
        combined_population = np.vstack((population, Sm))
        combined_fitness = np.array([func(*ind) for ind in combined_population])
        
        # Отбор лучших особей для новой популяции
        best_combined_indices = np.argsort(combined_fitness)[:population_size]
        population = combined_population[best_combined_indices]
        
        # Текущее лучшее решение
        current_best_fitness = combined_fitness[best_combined_indices[0]]
        current_best_position = population[0]

        # Уменьшение уровня мутации
        mutation_rate -= delta_mutation

        # Проверка критерия остановки (отсутствие улучшений)
        if abs(best_fitness - current_best_fitness) < tolerance:
            no_improvement_steps += 1
        else:
            no_improvement_steps = 0
            
        # Если долго нет улучшений - завершаем оптимизацию
        if no_improvement_steps >= tolerance_steps:
            history.append({
                'iteration': iteration+1,
                'x': best_position[0],
                'y': best_position[1],
                'f_value': best_fitness
            })
            converged = True
            message = "Оптимум найден"
            
            return history, converged, message

        # Обновление лучшего решения
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_position = current_best_position
        
        # Сохранение истории
        history.append({
            'iteration': iteration+1,
            'x': best_position[0],
            'y': best_position[1],
            'f_value': best_fitness
        })

    # Если вышли по количеству итераций
    converged = False
    message = "Достигнуто максимальное количество итераций"
    
    return history, converged, message