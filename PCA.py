import numpy as np

# Objective function: minimize f(x) = x^2 - 4x + 4
def objective_function(x):
    return x**2 - 4*x + 4

# Function to get neighbors (with boundary checks)
def get_neighbors(grid, i, j, neighborhood_size=3):
    rows, cols = grid.shape
    half = neighborhood_size // 2
    neighbors = []
    for x in range(i - half, i + half + 1):
        for y in range(j - half, j + half + 1):
            if 0 <= x < rows and 0 <= y < cols and not (x == i and y == j):
                neighbors.append(grid[x, y])
    return neighbors

# Main Parallel Cellular Algorithm
def parallel_cellular_algorithm(
    grid_size=(10, 10),
    search_space=(-10, 10),
    max_iterations=100,
    neighborhood_size=3
):
    rows, cols = grid_size
    # Step 1: Initialize population (random)
    grid = np.random.uniform(search_space[0], search_space[1], size=grid_size)
    fitness = objective_function(grid)

    for iteration in range(max_iterations):
        new_grid = np.copy(grid)

        # Step 2: Update all cells (parallel concept simulated)
        for i in range(rows):
            for j in range(cols):
                neighbors = get_neighbors(grid, i, j, neighborhood_size)
                neighbor_fitness = [objective_function(n) for n in neighbors]
                best_neighbor = neighbors[np.argmin(neighbor_fitness)]

                # Update rule: average between current and best neighbor
                new_grid[i, j] = (grid[i, j] + best_neighbor) / 2

        # Step 3: Evaluate new fitness
        grid = new_grid
        fitness = objective_function(grid)

        # Step 4: Track best solution
        best_index = np.unravel_index(np.argmin(fitness), fitness.shape)
        best_value = grid[best_index]
        best_fitness = fitness[best_index]

        # Optional: print progress
        if iteration % 10 == 0 or iteration == max_iterations - 1:
            print(f"Iteration {iteration+1}: Best x = {best_value:.4f}, f(x) = {best_fitness:.6f}")

    print("\nFinal Best Solution:")
    print(f"Best x = {best_value:.4f}, f(x) = {best_fitness:.6f}")

    return best_value, best_fitness


# Run the algorithm
if __name__ == "__main__":
    parallel_cellular_algorithm()
