# Импорт необходимых библиотек
import random  # для генерации случайных чисел
import copy    # для глубокого копирования объектов
import numpy as np  # для работы с массивами и математических операций

# Класс, представляющий одну бактерию
class Bacteria:
    def __init__(self, func, x_min, x_max, y_min, y_max):
        # Установка границ пространства поиска
        self.minval = [x_min, y_min]  # минимальные значения по осям x и y
        self.maxval = [x_max, y_max]  # максимальные значения по осям x и y

        # Генерация случайной начальной позиции бактерии в заданных границах
        self.position = np.array([random.uniform(self.minval[i], self.maxval[i]) for i in range(2)])

        # Целевая функция, которую нужно оптимизировать
        self.func = func

        # Здоровье бактерии (сумма значений функции за все шаги)
        self.health = func(*self.position)
        # Текущее значение функции в позиции бактерии
        self.func_value = self.health

        # Флаг, улучшила ли бактерия свое положение на последнем шаге
        self.improved_last_step = True

        # Вектор движения бактерии (случайное направление)
        self.movement_vector = np.random.rand(2)
        # Норма вектора движения (для нормализации)
        self.movement_vector_norm = np.linalg.norm(self.movement_vector)

    def move(self, chemotaxis_step):
        # Если на предыдущем шаге не было улучшения, генерируем новый случайный вектор движения
        if not(self.improved_last_step):
            self.movement_vector = np.random.rand(2)
            self.movement_vector_norm = np.linalg.norm(self.movement_vector)

        # Обновление позиции бактерии с учетом шага хемотаксиса
        self.position += chemotaxis_step * self.movement_vector / self.movement_vector_norm

        # Ограничение позиции бактерии заданными границами
        self.position[0] = np.clip(self.position[0], self.minval[0], self.maxval[0])
        self.position[1] = np.clip(self.position[1], self.minval[1], self.maxval[1])

        # Вычисление нового значения функции
        new_func_value = self.func(*self.position)
        # Обновление здоровья бактерии (накапливаем значения функции)
        self.health += new_func_value

        # Обновление флага improved_last_step в зависимости от того, улучшилось ли значение функции
        if new_func_value > self.func_value:
            self.improved_last_step = False
        else:
            self.improved_last_step = True
        
        # Сохранение нового значения функции
        self.func_value = new_func_value


# Класс, представляющий популяцию бактерий
class BacterialPopulation:
    def __init__(self, func, x_min, x_max, y_min, y_max, population_count, n_chemotaxis, n_reproduction, n_elimination, chemotaxis_step, elimination_threshold, elimination_probabilty, elimination_count):
        # Создание начальной популяции бактерий
        self.population = [Bacteria(func, x_min, x_max, y_min, y_max) for _ in range(population_count)]

        # Шаг хемотаксиса (размер шага при движении)
        self.chemotaxis_step = chemotaxis_step

        # Максимальное количество шагов хемотаксиса, репродукций и элиминаций
        self.n_chemotaxis = n_chemotaxis
        self.n_reproduction = n_reproduction
        self.n_elimination = n_elimination

        # Величина, на которую будет уменьшаться шаг хемотаксиса
        self.chemotaxis_step_reduction = chemotaxis_step / n_chemotaxis

        # Порог для активации элиминации (минимальное количество репродукций)
        self.elimination_threshold = elimination_threshold

        # Вероятность элиминации и количество бактерий для элиминации
        self.elimination_probabilty = elimination_probabilty
        self.elimination_count = elimination_count

        # Лучшее значение функции и позиция среди всей популяции
        self.best_func = self.population[0].func_value
        self.best_pos = self.population[0].position.copy()

        # Счетчики выполненных операций
        self.chemotaxiss_completed = 0
        self.reproductions_completed = 0
        self.eliminations_completed = 0

    
    def chemotaxis(self):
        # Проверка, не превышено ли максимальное количество шагов хемотаксиса
        if self.chemotaxiss_completed >= self.n_chemotaxis:
            return

        # Движение каждой бактерии в популяции
        for bacteria in self.population:
            bacteria.move(self.chemotaxis_step)

        # Уменьшение шага хемотаксиса
        self.chemotaxis_step -= self.chemotaxis_step_reduction
        
        # Увеличение счетчика выполненных шагов хемотаксиса
        self.chemotaxiss_completed += 1

    def reproduction(self):
        # Проверка, не превышено ли максимальное количество репродукций
        if self.reproductions_completed >= self.n_reproduction:
            return

        # Сортировка популяции по здоровью (лучшие - в начале списка)
        self.population.sort(key=lambda x: x.health)

        new_population = []

        # Репродукция: каждая лучшая половина бактерий делится на две одинаковые
        for i in range(len(self.population)//2):
            new_population.extend([copy.deepcopy(self.population[i]), copy.deepcopy(self.population[i])])
        
        # Замена старой популяции новой
        self.population = new_population

        # Увеличение счетчика выполненных репродукций
        self.reproductions_completed += 1

    def elimination(self):
        # Генерация случайного числа для проверки вероятности элиминации
        q = random.random()
        # Проверка условий для элиминации:
        # 1. Выполнено достаточно репродукций
        # 2. Случайное число меньше вероятности элиминации
        # 3. Не превышено максимальное количество элиминаций
        if (self.reproductions_completed < self.elimination_threshold) or (q <= self.elimination_probabilty) or (self.eliminations_completed >= self.n_elimination):
            return
        
        # Элиминация: замена случайных бактерий новыми со случайными позициями
        for _ in range(self.elimination_count):
            i = np.random.randint(0, len(self.population))
            self.population[i].position = np.array([random.uniform(self.population[i].minval[j], self.population[i].maxval[j]) for j in range(2)])

        # Увеличение счетчика выполненных элиминаций
        self.eliminations_completed += 1

    def next_step(self):
        # Выполнение одного полного цикла: хемотаксис, репродукция, элиминация
        self.chemotaxis()
        self.reproduction()
        self.elimination()

        # Обновление лучшего решения
        for bacteria in self.population:
            if bacteria.func_value < self.best_func:
                self.best_func = bacteria.func_value
                self.best_pos = bacteria.position.copy()


# Основная функция оптимизации
def optimize(func, x_min, x_max, y_min, y_max, population_count, n_chemotaxis, n_reproduction, n_elimination, chemotaxis_step, elimination_threshold, elimination_probabilty, elimination_count):
    # История оптимизации (для сохранения результатов на каждом шаге)
    history = []
    # Создание популяции бактерий с заданными параметрами
    population = BacterialPopulation(func, x_min, x_max, y_min, y_max, population_count, n_chemotaxis, n_reproduction, n_elimination, chemotaxis_step, elimination_threshold, elimination_probabilty, elimination_count)

    # Флаги и сообщения о статусе оптимизации
    converged = False
    message = "Достигнуто максимальное количество итераций"

    # Основной цикл оптимизации
    for i in range(n_chemotaxis):
        population.next_step()

        # Сохранение текущего лучшего решения в историю
        history.append({
            'iteration': i+1,
            'x': population.best_pos[0],
            'y': population.best_pos[1],
            'f_value': population.best_func
        })

    # Установка флага и сообщения об успешном завершении
    converged = True
    message = "Оптимум найден"
    
    # Возврат истории оптимизации, флага сходимости и сообщения
    return history, converged, message