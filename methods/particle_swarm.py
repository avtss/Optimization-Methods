import time
import numpy as np

class Particle:
    def __init__(self, swarm):
        self._position = self.getInitPosition(swarm)#начальная позиция

        self._localBestPosition = self._position[:]#сохраняем как лучшее решение

        self._localBestValue = swarm.getFuncValue(self._position)#считаем и запоминаем как лучшее значение

        self._velocity = self.getInitVelocity(swarm)#начальная скорость частицы


    @property
    def position(self):
        return self._position
    
    @property
    def velocity(self):
        return self._velocity
    
  #Создаёт случайную точку внутри диапазона
    def getInitPosition(self, swarm):
        return np.random.rand(swarm.dimension) * (swarm.maxvalues - swarm.minvalues) + swarm.minvalues
    
    #Задаёт начальную скорость 
    def getInitVelocity(self, swarm):
        minval = -(swarm.maxvalues - swarm.minvalues)
        maxval = swarm.maxvalues - swarm.minvalues

        return np.random.rand(swarm.dimension) * (maxval - minval) + minval
    

    def nextIteration(self, swarm):
        #векторы  инерции и тяготения к лучшим позициям
        random_currentPosition = np.random.rand(swarm.dimension)
        random_globalPosition = np.random.rand(swarm.dimension)
        
        #влияние локального и глобального лучшего решения
        velocityRatio = swarm.localVelocityRatio + swarm.globalVelocityRatio
        #Вычисляется коэффициент,используется для масштабирования всей скорости 
        under_sqrt = velocityRatio**2 - 4.0*velocityRatio
        if under_sqrt < 0:
            under_sqrt = 0  # или можно выбросить исключение, но обычно берут 0
        commonRatio = (2.0*swarm.currentVelocityRatio) / (np.abs(2.0 - velocityRatio - np.sqrt(under_sqrt)))

        newVelocity1 = commonRatio * self._velocity #инерция 
        newVelocity2 = commonRatio * swarm.localVelocityRatio * random_currentPosition * (self._localBestPosition - self._position)#притяжение к локальному лучшему
        newVelocity3 = commonRatio * swarm.globalVelocityRatio * random_globalPosition * (swarm.globalBestPosition - self._position)#притяжение к глобальному лучшему

        newVelocity = newVelocity1 + newVelocity2 + newVelocity3

        self._velocity = newVelocity

        self._position += self._velocity
        #считаем значение функции в новой точке
        funcValue = swarm.getFuncValue(self._position)
        #Если новая позиция лучше, она сохраняется как новое локальное лучшее
        if funcValue < self._localBestValue:
            self._localBestPosition = self._position[:]
            self._localBestValue = funcValue

class Swarm:
    def __init__(self, func, swarmsize, minvalues, maxvalues, currentVelocityRatio, localVelocityRatio, globalVelocityRatio, penaltyRatio):
        self._func = func
        self._swarmsize = swarmsize
        self._minvalues = np.array(minvalues[:])
        self._maxvalues = np.array(maxvalues[:])
        self._currentVelocityRatio = currentVelocityRatio #инерция
        self._localVelocityRatio = localVelocityRatio#локальное лучшее 
        self._globalVelocityRatio = globalVelocityRatio#глобальное лучшее
        self._globalBestValue = None
        self._globalBestPosition = None
        self._penaltyRatio = penaltyRatio #коэффиценты штрафов

        self._swarm = self.createSwarm()

    @property
    def func(self):
        return self._func

    @property
    def minvalues(self):
        return self._minvalues

    @property
    def maxvalues(self):
        return self._maxvalues

    @property
    def currentVelocityRatio(self):
        return self._currentVelocityRatio

    @property
    def localVelocityRatio(self):
        return self._localVelocityRatio

    @property
    def globalVelocityRatio(self):
        return self._globalVelocityRatio

    @property
    def globalBestPosition(self):
        return self._globalBestPosition

    @property
    def globalBestValue(self):
        return self._globalBestValue
    
    @property
    def penaltyRatio(self):
        return self._penaltyRatio
    
    @property
    def dimension(self):
        return len(self._minvalues)
    #Получить частицу по индексу
    def getParticle(self, index):
        return self._swarm[index]
    #Создание частиц в рое
    def createSwarm(self):
        return [Particle(self) for _ in range(self._swarmsize)]
    #за итерацию  обновляем все частицы
    def nextIteration(self):
        for particle in self._swarm:
            particle.nextIteration(self)
    
    def getFuncValue(self, position):
        result = self._func(*position)
        #Если это наилучшее значение из всех, сохраняется как глобальное
        if (self._globalBestValue is None) or (result < self._globalBestValue):
            self._globalBestValue = result
            self._globalBestPosition = position[:]

        return result + self.getPenalty(position)
    #если координата вне допустимого диапазона, добавляется штраф
    def getPenalty(self, position):
        penalty1 = sum([self._penaltyRatio*abs(coord-minval) for coord, minval in zip(position, self.minvalues) if coord < minval])
        penalty2 = sum([self._penaltyRatio*abs(coord-maxval) for coord, maxval in zip(position, self.maxvalues) if coord > maxval])

        return penalty1 + penalty2
    

def optimize(func, maxIter, swarmsize, bounds, currentVelocityRatio, localVelocityRatio, globalVelocityRatio, penaltyRatio,initial_positions = None, verbose=True):

    # Инициализация параметров
    history = []
    start_time = time.time()
    swarm = Swarm(func, swarmsize, bounds[:][0], bounds[:][1], currentVelocityRatio, localVelocityRatio, globalVelocityRatio, penaltyRatio)
    if initial_positions is not None:
        for i, pos in enumerate(initial_positions):
            if i < len(swarm._swarm):
                swarm._swarm[i]._position = np.array(pos)
                swarm._swarm[i]._localBestPosition = np.array(pos)
                swarm._swarm[i]._localBestValue = swarm.getFuncValue(pos)
    for i in range(maxIter):
        swarm.nextIteration()
        history.append({
            'iteration': i+1,
            'x': swarm.globalBestPosition[0],
            'y': swarm.globalBestPosition[1],
            'f_value': swarm.globalBestValue
        })
    
    if verbose:
        print(f"Время выполнения рой частиц: {time.time() - start_time:.2f} сек")
    # Формирование результата
    converged = True
    message = "Оптимум найден" if converged else "Достигнуто максимальное количество итераций"
    
    return history, converged, message