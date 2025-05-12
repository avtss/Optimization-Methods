import time
import numpy as np
from tabulate import tabulate
from methods import particle_swarm, bee, hybrid
from functions import functions

def test_algorithm(algorithm, func_name, bounds, params, n_runs=5):
    func = functions(func_name)
    errors = []
    times = []
    successes = 0
    
    for _ in range(n_runs):
        start_time = time.time()
        
        try:
            if algorithm == "PSO":
                history, converged, _ = particle_swarm.optimize(
                    func=func,
                    maxIter=params["max_iter"],
                    swarmsize=params["swarmsize"],
                    bounds=[bounds[0], bounds[1]],
                    currentVelocityRatio=params["velocity"],
                    localVelocityRatio=params["local_velocity"],
                    globalVelocityRatio=params["global_velocity"],
                    penaltyRatio=params["penalty"]
                )
            elif algorithm == "Bee":
                history, converged, _ = bee.optimize(
                    func=func,
                    maxiter=params["max_iter"],
                    scoutbee_count=params["scoutbees"],
                    selectedbee_count=params["selbees"],
                    bestbee_count=params["bestbees"],
                    bestsites_count=params["bestsites"],
                    selectedsites_count=params["selsites"],
                    radius=params["radius"],
                    koeff=params["koeff"],
                    tolerance=params["tolerance"],
                    globaltolerance=params["globaltolerance"],
                    x_min=bounds[0][0],
                    x_max=bounds[0][1],
                    y_min=bounds[1][0],
                    y_max=bounds[1][1]
                )
            elif algorithm == "Hybrid":
                history, converged, _ = hybrid.hybrid_optimize(
                    func=func,
                    maxiter=params["max_iter"],
                    swarmsize=params["swarmsize"],
                    bounds=[bounds[0], bounds[1]],
                    w=params["w"],
                    c1=params["c1"],
                    c2=params["c2"],
                    scoutbee_count=params["scoutbees"],
                    selectedbee_count=params["selbees"],
                    bestbee_count=params["bestbees"],
                    selectedsites_count=params["selsites"],
                    bestsites_count=params["bestsites"],
                    initial_radius=params["radius"],
                    koeff=params["koeff"],
                    tolerance=params["tolerance"],
                    globaltolerance=params["globaltolerance"]
                )
            
            elapsed_time = time.time() - start_time
            final_error = history[-1]['f_value']
            
            errors.append(final_error)
            times.append(elapsed_time)
            
            if final_error < 1e-3:
                successes += 1
                
        except Exception as e:
            print(f"Ошибка при выполнении {algorithm} на {func_name}: {str(e)}")
            errors.append(float('inf'))
            times.append(0)
    
    avg_error = np.mean(errors)
    avg_time = np.mean(times)
    success_rate = (successes / n_runs) * 100
    
    return {
        "Algorithm": algorithm,
        "Function": func_name,
        "Avg Error": avg_error,
        "Avg Time (s)": avg_time,
        "Success Rate (%)": success_rate
    }

# Параметры из вашего Dash-приложения
algorithm_params = {
    "PSO": {
        "max_iter": 100,
        "swarmsize": 50,
        "velocity": 0.7,
        "local_velocity": 1.5,
        "global_velocity": 1.5,
        "penalty": 10000
    },
    "Bee": {
        "max_iter": 100,
        "scoutbees": 20,
        "selbees": 15,
        "bestbees": 30,
        "bestsites": 2,
        "selsites": 3,
        "radius": 0.5,
        "koeff": 0.9,
        "tolerance": 5,
        "globaltolerance": 10
    },
    "Hybrid": {
        "max_iter": 100,
        "swarmsize": 50,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        "scoutbees": 20,
        "selbees": 15,
        "bestbees": 30,
        "selsites": 3,
        "bestsites": 2,
        "radius": 0.5,
        "koeff": 0.9,
        "tolerance": 5,
        "globaltolerance": 10
    }
}

test_functions = ["rosenbrock", "himmelblau"]
bounds = [[-5, 5], [-5, 5]]

# Запуск тестов
results = []
for func in test_functions:
    for algo in ["PSO", "Bee", "Hybrid"]:
        print(f"Testing {algo} on {func}...")
        res = test_algorithm(algo, func, bounds, algorithm_params[algo])
        results.append(res)
