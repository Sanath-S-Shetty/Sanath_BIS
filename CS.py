import random

# Problem data
weights = [2, 3, 6, 5]
values = [1, 4, 5, 6]
capacity = 6
n = len(weights)

# Parameters
population_size = 10
max_iterations = 100
abandon_prob = 0.25

def fitness(solution):
    total_weight = sum(w for w, s in zip(weights, solution) if s == 1)
    total_value = sum(v for v, s in zip(values, solution) if s == 1)
    if total_weight > capacity:
        return 0  # Penalize infeasible solutions
    return total_value

def random_solution():
    return [random.randint(0, 1) for _ in range(n)]

def generate_new_solution(current):
    # Flip a few bits randomly (mutation)
    new_solution = current[:]
    flip_count = random.randint(1, n // 2)
    indices = random.sample(range(n), flip_count)
    for idx in indices:
        new_solution[idx] = 1 - new_solution[idx]
    return new_solution

def kuku_search():
    # Initialize population
    population = [random_solution() for _ in range(population_size)]
    
    # Evaluate fitness
    fitnesses = [fitness(sol) for sol in population]
    
    best_solution = population[fitnesses.index(max(fitnesses))]
    best_fitness = max(fitnesses)
    
    for _ in range(max_iterations):
        new_population = []
        for sol in population:
            new_sol = generate_new_solution(sol)
            if fitness(new_sol) > fitness(sol):
                new_population.append(new_sol)
            else:
                new_population.append(sol)
        
        # Abandon worst solutions with probability abandon_prob
        for i in range(population_size):
            if random.random() < abandon_prob:
                new_population[i] = random_solution()
        
        population = new_population
        fitnesses = [fitness(sol) for sol in population]
        
        current_best = max(fitnesses)
        if current_best > best_fitness:
            best_fitness = current_best
            best_solution = population[fitnesses.index(current_best)]
    
    return best_solution, best_fitness

best_sol, best_val = kuku_search()
print(f"Best solution: {best_sol}")
print(f"Best value: {best_val}")
