import time
import numpy as np
import random
import math

class HybridBeeParticle:
    def __init__(self, func, x_min, x_max, y_min, y_max):
    
        self.minval = np.array([x_min, y_min])  # Преобразуем в numpy array
        self.maxval = np.array([x_max, y_max])  # Преобразуем в numpy array
        
        # Остальной код остается без изменений
        self.position = np.random.rand(2) * (self.maxval - self.minval) + self.minval
        self.fitness = func(*self.position)
        self.func = func
        self.velocity = np.random.rand(2) * (self.maxval - self.minval)
        self.best_position = self.position.copy()
        self.best_fitness = self.fitness
    
    def update_velocity(self, global_best_position, w, c1, c2):
        # PSO velocity update
        r1 = np.random.rand(2)
        r2 = np.random.rand(2)
        
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self):
        # Обновление позиции с учетом скорости (PSO)
        self.position += self.velocity
        
        # Проверка границ (Bee behavior)
        self.position = np.clip(self.position, self.minval, self.maxval)
        
        # Обновление фитнес-функции
        self.fitness = self.func(*self.position)
        
        # Обновление лучшей позиции
        if self.fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness
    
    def local_search(self, radius):
        # Локальный поиск как в пчелином алгоритме
        new_position = self.position + np.random.uniform(-radius, radius, size=2)
        new_position = np.clip(new_position, self.minval, self.maxval)
        new_fitness = self.func(*new_position)
        
        if new_fitness < self.fitness:
            self.position = new_position
            self.fitness = new_fitness
            if new_fitness < self.best_fitness:
                self.best_position = new_position.copy()
                self.best_fitness = new_fitness

class HybridSwarm:
    def __init__(self, func, swarmsize, bounds, w, c1, c2, scoutbee_count, selectedbee_count, bestbee_count, 
                 selectedsites_count, bestsites_count, initial_radius):
        # Параметры PSO
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter
        
        # Параметры Bee Algorithm
        self.scoutbee_count = scoutbee_count
        self.selectedbee_count = selectedbee_count
        self.bestbee_count = bestbee_count
        self.selectedsites_count = selectedsites_count
        self.bestsites_count = bestsites_count
        self.radius = initial_radius
        
        # Границы поиска
        self.minval = np.array([bounds[0][0], bounds[1][0]])
        self.maxval = np.array([bounds[0][1], bounds[1][1]])
        
        # Целевая функция
        self.func = func
        
        # Инициализация роя
        self.swarm = [HybridBeeParticle(func, 
                                      bounds[0][0], bounds[0][1],  # x_min, x_max
                                      bounds[1][0], bounds[1][1])  # y_min, y_max
                     for _ in range(swarmsize)]        
        # Лучшие позиции
        self.global_best_position = None
        self.global_best_fitness = math.inf
        self.update_global_best()
        
        # Лучшие участки (для пчелиной части)
        self.bestsites = []
        self.selectedsites = []
    
    def update_global_best(self):
        for particle in self.swarm:
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = particle.best_position.copy()
    
    def classify_sites(self):
        # Сортировка частиц по фитнесу
        self.swarm.sort(key=lambda x: x.fitness)
        
        # Выбор лучших участков
        self.bestsites = [self.swarm[0]]
        curr_index = 1
        
        while curr_index < len(self.swarm) and len(self.bestsites) < self.bestsites_count:
            particle = self.swarm[curr_index]
            # Проверяем, что частица находится достаточно далеко от уже выбранных лучших участков
            if all(np.linalg.norm(particle.position - best.position) > self.radius 
                   for best in self.bestsites):
                self.bestsites.append(particle)
            curr_index += 1
        
        # Выбор перспективных участков
        self.selectedsites = []
        while curr_index < len(self.swarm) and len(self.selectedsites) < self.selectedsites_count:
            particle = self.swarm[curr_index]
            # Проверяем, что частица не на лучших участках и достаточно далеко от других перспективных
            if (all(np.linalg.norm(particle.position - best.position) > self.radius 
                   for best in self.bestsites) and
                all(np.linalg.norm(particle.position - sel.position) > self.radius 
                   for sel in self.selectedsites)):
                self.selectedsites.append(particle)
            curr_index += 1
    
    def send_bees(self):
        # Отправляем пчел на лучшие участки
        bee_index = 0
        for best_particle in self.bestsites:
            for _ in range(self.bestbee_count):
                if bee_index >= len(self.swarm):
                    break
                self.swarm[bee_index].local_search(self.radius)
                bee_index += 1
        
        # Отправляем пчел на перспективные участки
        for sel_particle in self.selectedsites:
            for _ in range(self.selectedbee_count):
                if bee_index >= len(self.swarm):
                    break
                self.swarm[bee_index].local_search(self.radius)
                bee_index += 1
        
        # Оставшиеся пчелы действуют как разведчики
        for i in range(bee_index, len(self.swarm)):
            self.swarm[i].position = np.array([random.uniform(self.minval[i], self.maxval[i]) 
                                             for i in range(2)])
            self.swarm[i].fitness = self.func(*self.swarm[i].position)
            if self.swarm[i].fitness < self.swarm[i].best_fitness:
                self.swarm[i].best_position = self.swarm[i].position.copy()
                self.swarm[i].best_fitness = self.swarm[i].fitness
    
    def next_iteration(self):
        # PSO часть: обновление скоростей и позиций
        for particle in self.swarm:
            particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
            particle.update_position()
        
        self.update_global_best()
        
        # Bee часть: классификация участков и локальный поиск
        self.classify_sites()
        self.send_bees()
        
        # Обновляем глобальный лучший после локального поиска
        self.update_global_best()

def hybrid_optimize(func, maxiter, swarmsize, bounds, 
                   w=0.7, c1=1.5, c2=1.5, 
                   scoutbee_count=20, selectedbee_count=15, bestbee_count=30,
                   selectedsites_count=3, bestsites_count=2, initial_radius=0.5,
                   koeff=0.9, tolerance=5, globaltolerance=10,
                   verbose=True):  # Добавлен флаг verbose
    
    if len(bounds) != 2 or len(bounds[0]) != 2 or len(bounds[1]) != 2:
        raise ValueError("Bounds must be in format [[x_min, x_max], [y_min, y_max]]")
    
    history = []
    swarm = HybridSwarm(func, swarmsize, bounds, w, c1, c2, 
                       scoutbee_count, selectedbee_count, bestbee_count,
                       selectedsites_count, bestsites_count, initial_radius)
    
    best_value = math.inf
    tolerance_counter = 0
    converged = False
    message = "Достигнуто максимальное количество итераций"
    
    # Добавлен вывод стартовой информации
    if verbose:
        print("==============================================")
        print(f"Запуск гибридного алгоритма с параметрами:")
        print(f"Макс. итераций: {maxiter}")
        print(f"Размер роя: {swarmsize}")
        print(f"Параметры PSO: w={w}, c1={c1}, c2={c2}")
        print(f"Параметры Bees: радиус={initial_radius}, коэф. изменения={koeff}")
        print("==============================================")
    
    start_time = time.time()  # Замер времени выполнения
    
    for i in range(maxiter):
        swarm.next_iteration()
        
        # Вывод информации о текущей итерации
        
        # Логика обновления радиуса и проверки сходимости
        if abs(swarm.global_best_fitness - best_value) > 1e-5:
            best_value = swarm.global_best_fitness
            swarm.radius /= koeff
            tolerance_counter = 0
           
        else:
            tolerance_counter += 1
            if tolerance_counter >= tolerance:
                swarm.radius *= koeff
                tolerance_counter = 0
                globaltolerance -= 1
                if globaltolerance == 0:
                    converged = True
                    message = "Достигнут предел расширений"
                    break
        
        history.append({
            'iteration': i+1,
            'x': swarm.global_best_position[0],
            'y': swarm.global_best_position[1],
            'f_value': swarm.global_best_fitness
        })
    
    # Финальный вывод результатов
    if verbose:
        print(f"Время выполнения гибрид: {time.time() - start_time:.2f} сек")
    
    if not converged:
        converged = True
        message = "Оптимум найден"
    
    return history, converged, message