"""
Black Widow Optimization Algorithm (BWO) Implementation

Adapted from the spider-inspired optimization algorithm.
Reference: Hayyolalam, V., & Pourhaji Kazem, A. A. (2020). Black Widow Optimization Algorithm.
"""

import numpy as np
import random
from typing import List, Tuple, Callable, Dict, Any
import math


class BlackWidowOptimization:
    """Black Widow Optimization Algorithm."""
    
    def __init__(self, 
                 objective_func: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]],
                 population_size: int = 30,
                 procreation_rate: float = 0.8,
                 cannibalism_rate: float = 0.3,
                 mutation_rate: float = 0.1,
                 max_iterations: int = 100,
                 random_seed: int = None):
        """
        Initialize BWO parameters.
        
        Args:
            objective_func: Function to minimize (accepts 1D array, returns scalar).
            bounds: List of (lower, upper) bounds for each dimension.
            population_size: Number of spiders (solutions).
            procreation_rate: Rate of procreation (proportion of population).
            cannibalism_rate: Rate of cannibalism (weak spiders removed).
            mutation_rate: Rate of mutation.
            max_iterations: Maximum number of iterations.
            random_seed: Random seed for reproducibility.
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = population_size
        self.pr = procreation_rate
        self.cr = cannibalism_rate
        self.mr = mutation_rate
        self.max_iter = max_iterations
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Initialize population
        self.spiders = self._initialize_population()
        self.fitness = np.array([self.objective_func(spider) for spider in self.spiders])
        
        # Sort spiders by fitness
        self._sort_population()
        self.best_spider = self.spiders[0].copy()
        self.best_fitness = self.fitness[0]
        
        # History tracking
        self.history = {
            'best_fitness': [],
            'best_solution': [],
            'avg_fitness': [],
            'iterations': []
        }
    
    def _initialize_population(self) -> np.ndarray:
        """Randomly initialize spiders within bounds."""
        spiders = np.zeros((self.pop_size, self.dim))
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            spiders[:, i] = lower + (upper - lower) * np.random.rand(self.pop_size)
        return spiders
    
    def _sort_population(self) -> None:
        """Sort spiders by fitness (ascending, minimization)."""
        sorted_indices = np.argsort(self.fitness)
        self.spiders = self.spiders[sorted_indices]
        self.fitness = self.fitness[sorted_indices]
    
    def _procreate(self) -> np.ndarray:
        """Procreate new spiders through mating."""
        num_parents = int(self.pr * self.pop_size)
        parents = self.spiders[:num_parents]  # Best spiders as parents
        
        offspring = []
        for _ in range(self.pop_size):
            # Randomly select two parents
            i, j = np.random.choice(num_parents, 2, replace=False)
            parent1 = parents[i]
            parent2 = parents[j]
            
            # Crossover: blend crossover (BLX-α)
            alpha = 0.5
            child = np.zeros(self.dim)
            for d in range(self.dim):
                min_val = min(parent1[d], parent2[d])
                max_val = max(parent1[d], parent2[d])
                range_val = max_val - min_val
                
                lower = min_val - alpha * range_val
                upper = max_val + alpha * range_val
                
                child[d] = lower + (upper - lower) * np.random.rand()
            
            offspring.append(child)
        
        return np.array(offspring)
    
    def _mutate(self, spiders: np.ndarray) -> np.ndarray:
        """Apply mutation to spiders."""
        mutated = spiders.copy()
        num_mutations = int(self.mr * self.pop_size * self.dim)
        
        for _ in range(num_mutations):
            i = np.random.randint(self.pop_size)
            d = np.random.randint(self.dim)
            lower, upper = self.bounds[d]
            
            # Gaussian mutation
            sigma = 0.1 * (upper - lower)
            mutated[i, d] += np.random.normal(0, sigma)
            
            # Apply bounds
            if mutated[i, d] < lower:
                mutated[i, d] = lower
            if mutated[i, d] > upper:
                mutated[i, d] = upper
        
        return mutated
    
    def _cannibalize(self, spiders: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove weak spiders through cannibalism."""
        num_survivors = int((1 - self.cr) * len(spiders))
        
        if num_survivors < 1:
            num_survivors = 1
        
        # Keep the best spiders
        sorted_indices = np.argsort(fitness)
        survivors = spiders[sorted_indices[:num_survivors]]
        survivor_fitness = fitness[sorted_indices[:num_survivors]]
        
        # Generate new random spiders to replace cannibalized ones
        num_new = self.pop_size - num_survivors
        new_spiders = np.zeros((num_new, self.dim))
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            new_spiders[:, i] = lower + (upper - lower) * np.random.rand(num_new)
        
        new_fitness = np.array([self.objective_func(spider) for spider in new_spiders])
        
        # Combine survivors and new spiders
        combined_spiders = np.vstack([survivors, new_spiders])
        combined_fitness = np.concatenate([survivor_fitness, new_fitness])
        
        return combined_spiders, combined_fitness
    
    def optimize(self, verbose: bool = False) -> Dict[str, Any]:
        """Run the optimization algorithm."""
        for iteration in range(self.max_iter):
            # Procreation phase
            offspring = self._procreate()
            
            # Mutation phase
            offspring = self._mutate(offspring)
            
            # Evaluate offspring
            offspring_fitness = np.array([self.objective_func(spider) for spider in offspring])
            
            # Combine parents and offspring
            combined_spiders = np.vstack([self.spiders, offspring])
            combined_fitness = np.concatenate([self.fitness, offspring_fitness])
            
            # Cannibalism phase
            self.spiders, self.fitness = self._cannibalize(combined_spiders, combined_fitness)
            
            # Sort population
            self._sort_population()
            
            # Update global best
            if self.fitness[0] < self.best_fitness:
                self.best_fitness = self.fitness[0]
                self.best_spider = self.spiders[0].copy()
            
            # Record history
            self.history['best_fitness'].append(self.best_fitness)
            self.history['best_solution'].append(self.best_spider.copy())
            self.history['avg_fitness'].append(np.mean(self.fitness))
            self.history['iterations'].append(iteration)
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: Best = {self.best_fitness:.6f}, "
                      f"Avg = {np.mean(self.fitness):.6f}")
        
        return {
            'best_solution': self.best_spider,
            'best_fitness': self.best_fitness,
            'final_population': self.spiders,
            'final_fitness': self.fitness,
            'history': self.history
        }
    
    def optimize_parallel(self, num_runs: int = 10, verbose: bool = False) -> Dict[str, Any]:
        """Run multiple independent BWO optimizations and return the best result."""
        best_overall_fitness = float('inf')
        best_overall_solution = None
        all_results = []
        
        for run in range(num_runs):
            if verbose:
                print(f"Starting BWO run {run+1}/{num_runs}")
            
            # Reinitialize for each run
            self.spiders = self._initialize_population()
            self.fitness = np.array([self.objective_func(spider) for spider in self.spiders])
            self._sort_population()
            self.best_spider = self.spiders[0].copy()
            self.best_fitness = self.fitness[0]
            
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
    # Test with Ackley function
    def ackley(x):
        n = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(2 * math.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + math.e
    
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    bwo = BlackWidowOptimization(
        objective_func=ackley,
        bounds=bounds,
        population_size=30,
        procreation_rate=0.8,
        cannibalism_rate=0.3,
        mutation_rate=0.1,
        max_iterations=100,
        random_seed=42
    )
    
    result = bwo.optimize(verbose=True)
    print(f"\nBest solution: {result['best_solution']}")
    print(f"Best fitness: {result['best_fitness']}")
    
    # Parallel optimization example
    parallel_result = bwo.optimize_parallel(num_runs=5, verbose=True)
    print(f"\nParallel best fitness: {parallel_result['best_fitness']}")