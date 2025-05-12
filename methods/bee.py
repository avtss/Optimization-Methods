import random, math
import time

# Класс Bee представляет отдельную пчелу в алгоритме пчелиной колонии
class Bee:
    def __init__(self, func, x_min, x_max, y_min, y_max):
        # Минимальные и максимальные значения для координат x и y
        self.minval = [x_min, y_min]
        self.maxval = [x_max, y_max]
        
        # Инициализация случайной позиции пчелы в заданных границах
        self.position = [random.uniform(self.minval[i], self.maxval[i]) for i in range(2)]
        
        # Фитнес-функция (значение целевой функции в текущей позиции)
        self.fitness = 0.0
        
        # Целевая функция, которую нужно оптимизировать
        self.func = func
    
    # Вычисление значения фитнес-функции в текущей позиции
    def calcFitness(self):
        self.fitness = self.func(*self.position)
    
    # Проверка, находится ли пчела на другом участке, отличном от пчел в bee_list
    def otherPatch(self, bee_list, radius):
        if len(bee_list) == 0:
            return True
        
        self.calcFitness()
        for bee in bee_list:
            bee.calcFitness()
            pos = bee.getPosition()
            for i in range(2):
                # Если расстояние по любой координате превышает радиус, участки считаются разными
                if abs(self.position[i] - pos[i]) > radius:
                    return True
                
        return False
    
    # Получение текущей позиции пчелы
    def getPosition(self):
        return [val for val in self.position]
    
    # Перемещение пчелы в окрестность другой позиции (в пределах заданного радиуса) лучший или выбранный участок
    def goto(self, otherpos, radius):
        self.position = [otherpos[n] + random.uniform(-radius, radius) for n in range(len(otherpos))]
        self.checkPosition()
        self.calcFitness()
    
    # Случайное перемещение пчелы в пределах границ поиска(используется для разведчиков)
    def gotorandom(self):
        self.position = [random.uniform(self.minval[i], self.maxval[i]) for i in range(2)]
        self.checkPosition()
        self.calcFitness()
    
    # Проверка и корректировка позиции, чтобы она не выходила за границы поиска
    def checkPosition(self):
        for n in range(len(self.position)):
            if self.position[n] < self.minval[n]:
                self.position[n] = self.minval[n]
            elif self.position[n] > self.maxval[n]:
                self.position[n] = self.maxval[n]

# Класс Hive представляет всю пчелиную колонию и управляет ее поведением
class Hive:
    def __init__(self, scoutbee_count, selectedbee_count, bestbee_count, selectedsites_count, bestsites_count, radius, func, x_min, x_max, y_min, y_max):
        # Параметры алгоритма:
        self.scoutbee_count = scoutbee_count       # Количество пчел-разведчиков
        self.selectedbee_count = selectedbee_count # Количество пчел для выбранных участков
        self.bestbee_count = bestbee_count         # Количество пчел для лучших участков
        self.selectedsites_count = selectedsites_count # Количество выбранных участков
        self.bestsites_count = bestsites_count     # Количество лучших участков
        self.radius = radius                       # Радиус поиска вокруг участка
        self.func = func                           # Целевая функция
        
        # Лучшая найденная позиция и ее значение
        self.best_position = None
        self.best_fitness = math.inf
        
        # Создание роя пчел
        bee_count = scoutbee_count + selectedbee_count * selectedsites_count + bestbee_count * bestsites_count
        self.swarm = [Bee(func, x_min, x_max, y_min, y_max) for _ in range(bee_count)]
        
        # Инициализация списков лучших и выбранных участков
        self.bestsites = []
        self.selectedsites = []
        
        # Сортировка пчел по значению фитнес-функции и сохранение лучшего результата
        self.swarm.sort(key=lambda x: x.fitness)
        self.best_position = self.swarm[0].getPosition()
        self.best_fitness = self.swarm[0].fitness
    
    # Отправка пчел на указанный участок
    def sendBees(self, position, index, count):
        for i in range(count):
            if index == len(self.swarm):
                break
            
            bee = self.swarm[index]
            # Отправляем только тех пчел, которые еще не на участках
            if not(bee in self.bestsites) and not(bee in self.selectedsites):
                bee.goto(position, self.radius)
            index += 1
        
        return index 
    
    # Выполнение одной итерации алгоритма
    def nextIteration(self):
        # Пересчет фитнес-функции для всех пчел
        for bee in self.swarm:
            bee.calcFitness()
        
        # Сортировка пчел по значению фитнес-функции
        self.swarm.sort(key=lambda x: x.fitness)
        self.best_position = self.swarm[0].getPosition()
        self.best_fitness = self.swarm[0].fitness
        
        # Выбор лучших участков
        self.bestsites = [self.swarm[0]]
        self.selectedsites = []
        
        # Поиск лучших участков (непересекающихся в пределах радиуса)
        curr_index = 1
        while curr_index < len(self.swarm) and len(self.bestsites) < self.bestsites_count:
            bee = self.swarm[curr_index]
            if bee.otherPatch(self.bestsites, self.radius):
                self.bestsites.append(bee)
            curr_index += 1
        
        # Поиск выбранных участков (непересекающихся с лучшими и между собой)
        while curr_index < len(self.swarm) and len(self.selectedsites) < self.selectedsites_count:
            bee = self.swarm[curr_index]
            if bee.otherPatch(self.bestsites, self.radius) and bee.otherPatch(self.selectedsites, self.radius):
                self.selectedsites.append(bee)
            curr_index += 1
        
        # Отправка пчел на лучшие участки
        bee_index = 1
        for best_bee in self.bestsites:
            bee_index = self.sendBees(best_bee.getPosition(), bee_index, self.bestbee_count)
        
        # Отправка пчел на выбранные участки
        for sel_bee in self.selectedsites:
            bee_index = self.sendBees(sel_bee.getPosition(), bee_index, self.selectedbee_count)
        
        # Оставшиеся пчелы отправляются на случайный поиск
        for bee in self.swarm[bee_index:-1]:
            bee.gotorandom()  

# Основная функция оптимизации
def optimize(func, maxiter, scoutbee_count, selectedbee_count, bestbee_count, bestsites_count, selectedsites_count, radius, koeff, tolerance, globaltolerance, x_min, x_max, y_min, y_max,verbose=True):
       
    history = []
    start_time=time.time()  
    # Инициализация улья с заданными параметрами
    hive = Hive(scoutbee_count, selectedbee_count, bestbee_count, selectedsites_count, bestsites_count, radius, func, x_min, x_max, y_min, y_max)
    
    best_value = math.inf
    tolerance_counter = 0
    converged = False
    message = "Достигнуто максимальное количество итераций"
    
    # Основной цикл оптимизации
    for i in range(maxiter):
        hive.nextIteration()
        
        # Адаптация радиуса поиска на основе изменения лучшего значения
        if abs(hive.best_fitness - best_value) > 1e-5:
            best_value = hive.best_fitness
            hive.radius = hive.radius * koeff  # Уменьшаем радиус поиска
            tolerance_counter = 0
        else:
            tolerance_counter += 1
            if tolerance_counter >= tolerance:
                hive.radius = hive.radius / koeff  # Увеличиваем радиус поиска
                tolerance_counter = 0
                globaltolerance -= 1
                if globaltolerance == 0:
                    converged = True
                    message = "Достигнут предел расширений"
                    break #Решение не улучшалось после нескольких расширений радиуса
        
        # Сохранение истории изменений
        history.append({
            'iteration': i+1,
            'x': hive.best_position[0],
            'y': hive.best_position[1],
            'f_value': hive.best_fitness
        })


    
    if verbose:
        print(f"Время выполнения пчелиный: {time.time() - start_time:.2f} сек")
     
    converged = True
    message = "Оптимум найден" if converged else "Достигнуто максимальное количество итераций"
    
    return history, converged, message