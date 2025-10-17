import random
import math

class AntColony:
    def __init__(self, n, m, alpha, beta, rho, q0, dist_matrix):
        self.n = n
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.dist_matrix = dist_matrix
        self.pheromone = [[1.0 for _ in range(n)] for _ in range(n)]
        self.best_solution = None
        self.best_length = float('inf')

    def distance(self, i, j):
        return self.dist_matrix[i][j]

    def run(self, max_iter): # each ant constructs a tour
        for _ in range(max_iter):
            all_solutions = []
            for l in range(self.m):
                solution = self.construct_solution()
                all_solutions.append(solution)
            self.update_pheromone(all_solutions)
            self.update_best_solution(all_solutions)

    def construct_solution(self):
        S = list(range(self.n))
        start_city = random.choice(S)
        S.remove(start_city)
        tour = [start_city]
        current_city = start_city
        
        while S:
            next_city = self.choose_next_city(current_city, S)
            tour.append(next_city)
            current_city = next_city
            S.remove(next_city)
        
        return tour

    def choose_next_city(self, current_city, S):
        total_prob = 0
        prob = []
        for city in S:
            total_prob += (self.pheromone[current_city][city] ** self.alpha) * \
                          ((1.0 / self.distance(current_city, city)) ** self.beta)
        for city in S:
            prob.append((self.pheromone[current_city][city] ** self.alpha) *
                        ((1.0 / self.distance(current_city, city)) ** self.beta) / total_prob)
        return random.choices(S, prob)[0]

    def update_pheromone(self, solutions):
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1 - self.rho)
        
        for solution in solutions:
            length = self.calculate_tour_length(solution)
            for i in range(self.n - 1):
                self.pheromone[solution[i]][solution[i + 1]] += 1.0 / length
            self.pheromone[solution[-1]][solution[0]] += 1.0 / length

    def calculate_tour_length(self, solution):
        length = 0
        for i in range(self.n - 1):
            length += self.distance(solution[i], solution[i + 1])
        length += self.distance(solution[-1], solution[0])
        return length

    def update_best_solution(self, solutions):
        for solution in solutions:
            length = self.calculate_tour_length(solution)
            if length < self.best_length:
                self.best_solution = solution
                self.best_length = length

n = 5
m = 10
alpha = 1
beta = 2
rho = 0.1
q0 = 0.9
dist_matrix = [[0, 2, 2, 5, 7],
               [2, 0, 3, 4, 6],
               [2, 3, 0, 3, 4],
               [5, 4, 3, 0, 2],
               [7, 6, 4, 2, 0]]

colony = AntColony(n, m, alpha, beta, rho, q0, dist_matrix)
colony.run(100)

print("Best Solution:", colony.best_solution)
print("Best Length:", colony.best_length)