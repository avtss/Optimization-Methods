import numpy as np
import time
import matplotlib.pyplot as plt
from methods.genetic_algorithm import optimize as ga_optimize
from methods.particle_swarm import optimize as pso_optimize
from methods.hybrid import hybrid_optimize
from functions import functions

# Конфигурация тестовых функций
global_minima = {
    "rosenbrock": ((1, 1), 0),
    "himmelblau": ((3.0, 2.0), 0),
    "isom": ((np.pi, np.pi), -1),
    "sphere": ((0, 0), 0),
    "rastrigin": ((0, 0), 0),
    "bukin": ((-10, 1), 0)
}

test_config = {
    "rosenbrock": {"bounds": [(-5, 5), (-5, 5)]},
    "himmelblau": {"bounds": [(-5, 5), (-5, 5)]},
    "isom": {"bounds": [(0, 6), (0, 6)]},
    "sphere": {"bounds": [(-5, 5), (-5, 5)]},
    "rastrigin": {"bounds": [(-5.12, 5.12), (-5.12, 5.12)]},
    "bukin": {"bounds": [(-15, -5), (-3, 3)]}
}

def compute_error(f, found_xy, global_min):
    try:
        if any(np.isnan(found_xy)) or any(np.isinf(found_xy)):
            return np.inf
        
        calculated = f(*found_xy)
        
        if np.isnan(calculated) or np.isinf(calculated):
            return np.inf
            
        return abs(calculated - global_min)
    except:
        return np.inf

def run_experiment(function_name, bounds, n_runs=10):
    f = functions(function_name)
    true_min, true_f = global_minima[function_name]

    results = {}
    for method_name, method in {
        'GA': ga_optimize,
        'PSO': pso_optimize,
        'Hybrid': hybrid_optimize,
    }.items():
        times = []
        errors = []
        for _ in range(n_runs):
            try:
                t0 = time.time()
                if method_name == 'GA':
                    history, _, _, _ = method(
                        objective_func=f,
                        bounds=bounds,
                        population_size=50,
                        max_iter=100,
                        verbose=False
                    )
                    best = min(history, key=lambda h: h['f_value'])
                elif method_name == 'PSO':
                    history, _, _ = method(
                        func=f,
                        maxIter=100,
                        swarmsize=50,
                        bounds=bounds,
                        currentVelocityRatio=0.5,
                        localVelocityRatio=2.0,
                        globalVelocityRatio=2.0,
                        penaltyRatio=10,
                        verbose=False
                    )
                    best = min(history, key=lambda h: h['f_value'])
                else:
                    history, _, _ = method(
                        func=f,
                        bounds=bounds,
                        ga_population_size=50,
                        ga_max_iter=50,
                        pso_swarmsize=30,
                        pso_max_iter=50,
                        pso_current_velocity=0.5,
                        pso_local_ratio=2.0,
                        pso_global_ratio=2.0,
                        pso_penalty=10,
                    )
                    best = min(history, key=lambda h: h['f_value'])
                
                t1 = time.time()
                error = compute_error(f, (best['x'], best['y']), true_f)
                
                if np.isnan(error) or np.isinf(error):
                    error = np.inf
                
                times.append(t1-t0)
                errors.append(error)
                
            except Exception as e:
                print(f"Ошибка в {method_name}: {str(e)}")
                errors.append(np.inf)
                times.append(np.inf)
        
        clean_errors = [e for e in errors if np.isfinite(e)]
        clean_times = [t for t in times if np.isfinite(t)]
        
        mean_time = np.mean(clean_times) if clean_times else np.inf
        std_time = np.std(clean_times) if clean_times else 0
        mean_error = np.mean(clean_errors) if clean_errors else np.inf
        std_error = np.std(clean_errors) if clean_errors else 0
        
        results[method_name] = {
            "mean_time": mean_time,
            "mean_error": mean_error,
        }
    return results

def plot_function_results(results, function_name):
    methods = list(results.keys())
    times = [res['mean_time'] for res in results.values()]
    errors = [res['mean_error'] for res in results.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График времени выполнения
    bars = ax1.bar(methods, times, color=['#2ecc71', '#3498db', '#9b59b6'])
    ax1.set_title(f'Среднее время выполнения\n({function_name})')
    ax1.set_ylabel('Секунды')
    ax1.bar_label(bars, fmt='%.3f', padding=3)
    
    # График ошибок
    bars = ax2.bar(methods, errors, color=['#e74c3c', '#f1c40f', '#34495e'])
    ax2.set_title(f'Средняя ошибка\n({function_name})')
    ax2.set_ylabel('Ошибка (log scale)')
    ax2.set_yscale('log')
    ax2.bar_label(bars, fmt='%.2e', padding=3)
    
    plt.tight_layout()
    plt.show()

def run_comparative_study(n_runs=10):
    all_results = {}
    
    for func_name in test_config.keys():
        print(f"\n=== Тестирование функции {func_name.upper()} ===")
        bounds = test_config[func_name]["bounds"]
        results = run_experiment(func_name, bounds, n_runs)
        all_results[func_name] = results
        plot_function_results(results, func_name)
    
    # Сводная таблица результатов
    print("\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
    for func_name in test_config.keys():
        print(f"\nФункция: {func_name}")
        for method, res in all_results[func_name].items():
            print(f"{method}:")
            print(f"  Время: {res['mean_time']:.4f} сек")
            print(f"  Ошибка: {res['mean_error']:.4e}")

if __name__ == "__main__":
    run_comparative_study(n_runs=10)