import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Objective Function (Problem) ---
# We will use the Sphere function, as mentioned in the slides 
# f(x) = sum(x_i^2). The global minimum is 0 at x_i = 0.
def sphere_function(position):
    """Calculates the Sphere function value for a given position."""
    return np.sum(position**2)

# --- 2. Grey Wolf Optimizer (GWO) Function ---
def grey_wolf_optimizer(objective_func, lb, ub, dim, n_wolves, max_iter):
   
    
    # [cite: 183] 1. Initialize population of wolves
    # Alpha, Beta, and Delta wolf positions and scores
    # [cite: 171, 172, 173]
    Alpha_pos = np.zeros(dim)
    Alpha_score = float('inf')  # We are minimizing, so start with infinity
    
    Beta_pos = np.zeros(dim)
    Beta_score = float('inf')
    
    Delta_pos = np.zeros(dim)
    Delta_score = float('inf')
    
    # Initialize the positions of all wolves (including Omegas)
    # [cite: 176] Wolves = candidate solutions
    Positions = np.random.uniform(lb, ub, (n_wolves, dim))
    
    convergence_curve = np.zeros(max_iter)

    # --- Start Main Loop ---
    # [cite: 188] 6. Repeat until max iterations
    for t in range(max_iter):
        
        #  2. Evaluate fitness of each wolf
        for i in range(n_wolves):
            # Handle boundary constraints
            #  5. Handle boundaries
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            
            # Calculate fitness
            fitness = objective_func(Positions[i, :])
            
            #  3. Identify alpha, beta, and delta wolves
            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                # This wolf is the new best (Alpha)
                Delta_score = Beta_score  # Old Beta becomes Delta
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Old Alpha becomes Beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness  # New Alpha
                Alpha_pos = Positions[i, :].copy()
            elif fitness < Beta_score:
                # This wolf is the new second best (Beta)
                Delta_score = Beta_score  # Old Beta becomes Delta
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  # New Beta
                Beta_pos = Positions[i, :].copy()
            elif fitness < Delta_score:
                # This wolf is the new third best (Delta)
                Delta_score = fitness  # New Delta
                Delta_pos = Positions[i, :].copy()

        # Update 'a' parameter 
        # 'a' decreases linearly from 2 to 0
        a = 2 - t * (2 / max_iter)
        
        # [cite: 186] 4. Update positions of wolves (Omega wolves)
        for i in range(n_wolves):
            # --- Update based on Alpha wolf ---
            r1, r2 = np.random.rand(dim), np.random.rand(dim)  # [cite: 202, 208]
            A1 = 2 * a * r1 - a  # Equation [cite: 205]
            C1 = 2 * r2  # Equation [cite: 201]
            D_alpha = np.abs(C1 * Alpha_pos - Positions[i, :])  # Equation [cite: 197]
            X1 = Alpha_pos - A1 * D_alpha  # Equation [cite: 203]
            
            # --- Update based on Beta wolf ---
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * Beta_pos - Positions[i, :])
            X2 = Beta_pos - A2 * D_beta
            
            # --- Update based on Delta wolf ---
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * Delta_pos - Positions[i, :])
            X3 = Delta_pos - A3 * D_delta
            
            # New position is the average of the three 
            Positions[i, :] = (X1 + X2 + X3) / 3
            
        # Store the best fitness value of this iteration [cite: 211]
        convergence_curve[t] = Alpha_score
        
        if (t + 1) % 10 == 0:
            print(f"Iteration {t+1}, Best Fitness (Alpha Score): {Alpha_score}")

    #  7. Return alpha wolf as best solution
    return Alpha_pos, Alpha_score, convergence_curve

# --- 3. Set Parameters and Run the Algorithm ---

# Problem parameters
DIMS = 10         # Dimensions
LOWER_BOUND = -10 # Lower search space bound
UPPER_BOUND = 10  # Upper search space bound

# GWO parameters
N_WOLVES = 30     # Number of wolves
MAX_ITER = 100    # Maximum iterations

print("--- Grey Wolf Optimizer (GWO) Running ---")
print(f"Minimizing {DIMS}-D Sphere function")
print(f"Parameters: {N_WOLVES} wolves, {MAX_ITER} iterations")
print("-" * 40)

# Run GWO
best_position, best_score, convergence = grey_wolf_optimizer(
    sphere_function, LOWER_BOUND, UPPER_BOUND, DIMS, N_WOLVES, MAX_ITER
)

print("-" * 40)
print("--- Results ---")
print(f"Best Solution (Alpha Position): {best_position}")
print(f"Best Fitness (Alpha Score): {best_score}")
print(f"Solution should be close to 0.0")

# --- 4. Plot Convergence Curve ---
#  This visualizes the optimization process
plt.figure(figsize=(10, 6))
plt.plot(convergence, label='GWO Convergence', color='blue')
plt.title('Convergence Curve of Grey Wolf Optimizer', fontsize=16)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Best Fitness Value', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--')
plt.savefig('gwo_convergence_curve.png')

print("\nConvergence curve plot saved as 'gwo_convergence_curve.png'")