"""
Cuckoo Optimization Algorithm (COA) Implementation

Adapted from the original cuckoo search algorithm with Levy flights.
Reference: Yang, X. S., & Deb, S. (2009). Cuckoo search via Lévy flights.
"""

import numpy as np
import random
from typing import List, Tuple, Callable, Dict, Any
import math


class CuckooOptimizationAlgorithm:
    """Cuckoo Search Optimization Algorithm with Lévy flights."""
    
    def __init__(self, 
                 objective_func: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]],
                 population_size: int = 25,
                 nest_abandonment_prob: float = 0.25,
                 step_size_alpha: float = 0.01,
                 levy_beta: float = 1.5,
                 max_iterations: int = 100,
                 random_seed: int = None):
        """
        Initialize COA parameters.
        
        Args:
            objective_func: Function to minimize (accepts 1D array, returns scalar).
            bounds: List of (lower, upper) bounds for each dimension.
            population_size: Number of nests (solutions).
            nest_abandonment_prob: Probability of abandoning a nest (discovery rate).
            step_size_alpha: Step size scaling factor.
            levy_beta: Lévy flight parameter (1 < beta <= 3).
            max_iterations: Maximum number of iterations.
            random_seed: Random seed for reproducibility.
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = population_size
        self.pa = nest_abandonment_prob
        self.alpha = step_size_alpha
        self.beta = levy_beta
        self.max_iter = max_iterations
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Initialize population
        self.nests = self._initialize_population()
        self.fitness = np.array([self.objective_func(nest) for nest in self.nests])
        self.best_nest = self.nests[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)
        
        # History tracking
        self.history = {
            'best_fitness': [],
            'best_solution': [],
            'avg_fitness': [],
            'iterations': []
        }
    
    def _initialize_population(self) -> np.ndarray:
        """Randomly initialize nests within bounds."""
        nests = np.zeros((self.pop_size, self.dim))
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            nests[:, i] = lower + (upper - lower) * np.random.rand(self.pop_size)
        return nests
    
    def _levy_flight(self, size: int = None) -> float:
        """Generate Lévy flight step using Mantegna's algorithm."""
        if size is None:
            size = self.dim
        
        sigma = (math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2) /
                 (math.gamma((1 + self.beta) / 2) * self.beta * 
                  2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        
        step = u / (np.abs(v) ** (1 / self.beta))
        return step
    
    def _get_new_cuckoo(self, idx: int) -> np.ndarray:
        """Generate new cuckoo via Lévy flight."""
        nest = self.nests[idx].copy()
        step = self._levy_flight()
        
        # Scale step by alpha and problem scale
        scale = 0.01 * (self.bounds[:, 1] - self.bounds[:, 0])
        new_nest = nest + self.alpha * step * scale
        
        # Apply bounds
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            if new_nest[i] < lower:
                new_nest[i] = lower
            if new_nest[i] > upper:
                new_nest[i] = upper
        
        return new_nest
    
    def _abandon_nests(self) -> None:
        """Abandon some nests and build new ones."""
        # Identify nests to abandon
        abandon_mask = np.random.rand(self.pop_size) < self.pa
        
        for i in range(self.pop_size):
            if abandon_mask[i]:
                # Generate new nest via random walk
                j, k = np.random.choice(self.pop_size, 2, replace=False)
                epsilon = np.random.rand(self.dim)
                new_nest = self.nests[i] + epsilon * (self.nests[j] - self.nests[k])
                
                # Apply bounds
                for d in range(self.dim):
                    lower, upper = self.bounds[d]
                    if new_nest[d] < lower:
                        new_nest[d] = lower
                    if new_nest[d] > upper:
                        new_nest[d] = upper
                
                self.nests[i] = new_nest
                self.fitness[i] = self.objective_func(new_nest)
    
    def optimize(self, verbose: bool = False) -> Dict[str, Any]:
        """Run the optimization algorithm."""
        for iteration in range(self.max_iter):
            # Generate new cuckoo
            i = np.random.randint(self.pop_size)
            new_nest = self._get_new_cuckoo(i)
            new_fitness = self.objective_func(new_nest)
            
            # Replace if better
            if new_fitness < self.fitness[i]:
                self.nests[i] = new_nest
                self.fitness[i] = new_fitness
            
            # Abandon worst nests
            self._abandon_nests()
            
            # Update global best
            current_best_idx = np.argmin(self.fitness)
            current_best_fitness = self.fitness[current_best_idx]
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_nest = self.nests[current_best_idx].copy()
            
            # Record history
            self.history['best_fitness'].append(self.best_fitness)
            self.history['best_solution'].append(self.best_nest.copy())
            self.history['avg_fitness'].append(np.mean(self.fitness))
            self.history['iterations'].append(iteration)
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: Best = {self.best_fitness:.6f}, "
                      f"Avg = {np.mean(self.fitness):.6f}")
        
        return {
            'best_solution': self.best_nest,
            'best_fitness': self.best_fitness,
            'final_population': self.nests,
            'final_fitness': self.fitness,
            'history': self.history
        }
    
    def optimize_parallel(self, num_runs: int = 10, verbose: bool = False) -> Dict[str, Any]:
        """Run multiple independent COA optimizations and return the best result."""
        best_overall_fitness = float('inf')
        best_overall_solution = None
        all_results = []
        
        for run in range(num_runs):
            if verbose:
                print(f"Starting COA run {run+1}/{num_runs}")
            
            # Reinitialize for each run
            self.nests = self._initialize_population()
            self.fitness = np.array([self.objective_func(nest) for nest in self.nests])
            self.best_nest = self.nests[np.argmin(self.fitness)]
            self.best_fitness = np.min(self.fitness)
            
            # Run optimization
            result = self.optimize(verbose=False)
            all_results.append(result)
            
            if result['best_fitness'] < best_overall_fitness:
                best_overall_fitness = result['best_fitness']
                best_overall_solution = result['best_solution']
        
        return {
            'best_solution': best_overall_solution,
            'best_fitness': best_overall_fitness,
            'num_runs': num_runs,
            'all_results': all_results
        }


# Example usage
if __name__ == "__main__":
    # Test with sphere function
    def sphere(x):
        return np.sum(x ** 2)
    
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    coa = CuckooOptimizationAlgorithm(
        objective_func=sphere,
        bounds=bounds,
        population_size=30,
        max_iterations=100,
        random_seed=42
    )
    
    result = coa.optimize(verbose=True)
    print(f"\nBest solution: {result['best_solution']}")
    print(f"Best fitness: {result['best_fitness']}")
    
    # Parallel optimization example
    parallel_result = coa.optimize_parallel(num_runs=5, verbose=True)
    print(f"\nParallel best fitness: {parallel_result['best_fitness']}")