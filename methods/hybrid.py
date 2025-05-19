# methods/hybrid.py
import numpy as np
from .genetic_algorithm import optimize as ga_optimize
from .particle_swarm import optimize as pso_optimize

def hybrid_optimize(
    func,
    bounds,
    ga_population_size=50,
    ga_max_iter=50,
    pso_swarmsize=30,
    pso_max_iter=30,
    pso_current_velocity=0.5,
    pso_local_ratio=1.0,
    pso_global_ratio=1.0,
    pso_penalty=10
):
    # Запуск GA
    ga_history, ga_converged, ga_message, ga_population = ga_optimize(
        objective_func=func,
        bounds=bounds,
        population_size=ga_population_size,
        max_iter=ga_max_iter
    )
    
    # Фильтрация NaN из GA
    valid_solutions = [ind for ind in ga_population if not np.isnan(ind).any()]
    if not valid_solutions:
        return [], False, "GA не нашел допустимых решений"
    
    # Выбор лучших решений
    sorted_solutions = sorted(valid_solutions, key=lambda x: func(x[0], x[1]))
    initial_positions = sorted_solutions[:pso_swarmsize]
    
    # Запуск PSO
    pso_history, pso_converged, pso_message = pso_optimize(
        func=func,
        maxIter=pso_max_iter,
        swarmsize=pso_swarmsize,
        bounds=bounds,
        currentVelocityRatio=0.5,  # Пример значения
        localVelocityRatio=1.0,
        globalVelocityRatio=1.0,
        penaltyRatio=10,
        initial_positions=initial_positions 
    )
    # Объединение истории
    combined_history = ga_history.copy()
    last_iter = ga_history[-1]['iteration'] if ga_history else 0
    
    for pso_point in pso_history:
        combined_history.append({
            'iteration': last_iter + pso_point['iteration'],
            'x': pso_point['x'],
            'y': pso_point['y'],
            'f_value': pso_point['f_value']
        })
    
    return combined_history, pso_converged, f"GA: {ga_message}, PSO: {pso_message}"