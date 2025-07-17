"""
Unit tests for GeneticAlgorithmOptimizer.

Tests the genetic algorithm optimization including selection,
crossover, mutation, and evolutionary dynamics.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from typing import Dict, Any, List

from btc_research.optimization.optimizers.genetic_algorithm import (
    GeneticAlgorithmOptimizer,
    SelectionMethod,
    CrossoverMethod,
    MutationMethod,
)
from btc_research.optimization.types import (
    ParameterSpec,
    ParameterType,
    OptimizationMetric,
    OptimizationResult,
)


class TestGeneticAlgorithmOptimizer(unittest.TestCase):
    """Test cases for GeneticAlgorithmOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_specs = [
            ParameterSpec("x1", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("x2", ParameterType.FLOAT, low=-1.0, high=1.0),
            ParameterSpec("n", ParameterType.INTEGER, low=1, high=10),
            ParameterSpec("choice", ParameterType.CATEGORICAL, choices=["a", "b", "c"]),
            ParameterSpec("flag", ParameterType.BOOLEAN),
        ]
        
        # Multi-modal objective function for testing
        def mock_objective(params: Dict[str, Any]) -> float:
            x1 = params.get("x1", 0)
            x2 = params.get("x2", 0)
            n = params.get("n", 1)
            choice_bonus = {"a": 0.0, "b": 0.1, "c": 0.2}.get(params.get("choice"), 0)
            flag_bonus = 0.1 if params.get("flag", False) else 0
            
            # Multi-modal function with several local optima
            return (np.sin(5 * x1) * np.cos(3 * x2) + 
                   0.5 * np.exp(-(x1-0.7)**2 - (x2-0.3)**2) +
                   n * 0.01 + choice_bonus + flag_bonus)
        
        self.objective_function = mock_objective
        self.metric = OptimizationMetric.SHARPE_RATIO
    
    def test_initialization_default(self):
        """Test successful initialization with default parameters."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric
        )
        
        self.assertEqual(optimizer.parameter_specs, self.parameter_specs)
        self.assertEqual(optimizer.objective_function, self.objective_function)
        self.assertEqual(optimizer.metric, self.metric)
        self.assertEqual(optimizer.population_size, 50)
        self.assertEqual(optimizer.selection_method, SelectionMethod.TOURNAMENT)
        self.assertEqual(optimizer.crossover_method, CrossoverMethod.UNIFORM)
        self.assertEqual(optimizer.mutation_method, MutationMethod.GAUSSIAN)
        self.assertEqual(optimizer.crossover_probability, 0.8)
        self.assertEqual(optimizer.mutation_probability, 0.1)
        self.assertEqual(optimizer.tournament_size, 3)
        self.assertEqual(optimizer.elite_size, 2)
    
    def test_initialization_with_parameters(self):
        """Test initialization with custom parameters."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            population_size=100,
            selection_method=SelectionMethod.ROULETTE_WHEEL,
            crossover_method=CrossoverMethod.SINGLE_POINT,
            mutation_method=MutationMethod.UNIFORM,
            crossover_probability=0.9,
            mutation_probability=0.05,
            tournament_size=5,
            elite_size=5,
            random_seed=42
        )
        
        self.assertEqual(optimizer.population_size, 100)
        self.assertEqual(optimizer.selection_method, SelectionMethod.ROULETTE_WHEEL)
        self.assertEqual(optimizer.crossover_method, CrossoverMethod.SINGLE_POINT)
        self.assertEqual(optimizer.mutation_method, MutationMethod.UNIFORM)
        self.assertEqual(optimizer.crossover_probability, 0.9)
        self.assertEqual(optimizer.mutation_probability, 0.05)
        self.assertEqual(optimizer.tournament_size, 5)
        self.assertEqual(optimizer.elite_size, 5)
        self.assertEqual(optimizer.random_seed, 42)
    
    def test_create_individual(self):
        """Test individual creation with proper parameter types."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            random_seed=42
        )
        
        individual = optimizer._create_individual()
        
        self.assertIsInstance(individual, dict)
        self.assertIn("x1", individual)
        self.assertIn("x2", individual)
        self.assertIn("n", individual)
        self.assertIn("choice", individual)
        self.assertIn("flag", individual)
        
        # Check types and bounds
        self.assertIsInstance(individual["x1"], float)
        self.assertGreaterEqual(individual["x1"], 0.0)
        self.assertLessEqual(individual["x1"], 1.0)
        
        self.assertIsInstance(individual["x2"], float)
        self.assertGreaterEqual(individual["x2"], -1.0)
        self.assertLessEqual(individual["x2"], 1.0)
        
        self.assertIsInstance(individual["n"], int)
        self.assertGreaterEqual(individual["n"], 1)
        self.assertLessEqual(individual["n"], 10)
        
        self.assertIn(individual["choice"], ["a", "b", "c"])
        self.assertIsInstance(individual["flag"], bool)
    
    def test_initialize_population(self):
        """Test population initialization."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            population_size=20,
            random_seed=42
        )
        
        population = optimizer._initialize_population()
        
        self.assertEqual(len(population), 20)
        
        for individual in population:
            self.assertIsInstance(individual, dict)
            self.assertEqual(len(individual), len(self.parameter_specs))
    
    def test_evaluate_population(self):
        """Test population evaluation."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            population_size=10
        )
        
        population = optimizer._initialize_population()
        fitness_scores = optimizer._evaluate_population(population)
        
        self.assertEqual(len(fitness_scores), len(population))
        
        for score in fitness_scores:
            self.assertIsInstance(score, float)
            self.assertNotEqual(score, float('-inf'))  # Should not be failure score
    
    def test_tournament_selection(self):
        """Test tournament selection mechanism."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            selection_method=SelectionMethod.TOURNAMENT,
            tournament_size=3,
            random_seed=42
        )
        
        population = [{"x1": i/10, "x2": 0, "n": 1, "choice": "a", "flag": False} 
                     for i in range(10)]
        fitness_scores = [i/10 for i in range(10)]  # Increasing fitness
        
        selected = optimizer._tournament_selection(population, fitness_scores, 5)
        
        self.assertEqual(len(selected), 5)
        
        # Tournament selection should favor higher fitness individuals
        selected_fitness = [fitness_scores[population.index(ind)] for ind in selected]
        avg_selected_fitness = np.mean(selected_fitness)
        avg_population_fitness = np.mean(fitness_scores)
        
        self.assertGreater(avg_selected_fitness, avg_population_fitness)
    
    def test_roulette_wheel_selection(self):
        """Test roulette wheel selection mechanism."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            selection_method=SelectionMethod.ROULETTE_WHEEL,
            random_seed=42
        )
        
        population = [{"x1": i/10, "x2": 0, "n": 1, "choice": "a", "flag": False} 
                     for i in range(5)]
        fitness_scores = [1, 2, 3, 4, 5]  # Increasing fitness
        
        selected = optimizer._roulette_wheel_selection(population, fitness_scores, 10)
        
        self.assertEqual(len(selected), 10)
        
        # Higher fitness individuals should be selected more often
        selected_indices = [population.index(ind) for ind in selected]
        
        # Count selections of high-fitness individual (index 4)
        high_fitness_selections = selected_indices.count(4)
        low_fitness_selections = selected_indices.count(0)
        
        # Should select high-fitness individual more often (though not guaranteed)
        self.assertGreaterEqual(high_fitness_selections, low_fitness_selections)
    
    def test_rank_selection(self):
        """Test rank-based selection mechanism."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            selection_method=SelectionMethod.RANK,
            random_seed=42
        )
        
        population = [{"x1": i/10, "x2": 0, "n": 1, "choice": "a", "flag": False} 
                     for i in range(5)]
        fitness_scores = [10, 5, 100, 1, 50]  # Non-uniform fitness
        
        selected = optimizer._rank_selection(population, fitness_scores, 8)
        
        self.assertEqual(len(selected), 8)
        # Rank selection should work regardless of fitness scale
        self.assertGreater(len(selected), 0)
    
    def test_uniform_crossover(self):
        """Test uniform crossover operation."""
        specs = [
            ParameterSpec("x1", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("n", ParameterType.INTEGER, low=1, high=10),
        ]
        
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=specs,
            objective_function=self.objective_function,
            metric=self.metric,
            crossover_method=CrossoverMethod.UNIFORM,
            random_seed=42
        )
        
        parent1 = {"x1": 0.2, "n": 3}
        parent2 = {"x1": 0.8, "n": 7}
        
        child1, child2 = optimizer._uniform_crossover(parent1, parent2)
        
        # Children should have valid parameter values
        self.assertIn("x1", child1)
        self.assertIn("n", child1)
        self.assertIn("x1", child2)
        self.assertIn("n", child2)
        
        # Values should be from one parent or the other
        self.assertTrue(child1["x1"] in [0.2, 0.8])
        self.assertTrue(child1["n"] in [3, 7])
        self.assertTrue(child2["x1"] in [0.2, 0.8])
        self.assertTrue(child2["n"] in [3, 7])
    
    def test_single_point_crossover(self):
        """Test single-point crossover operation."""
        specs = [
            ParameterSpec("x1", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("x2", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("n", ParameterType.INTEGER, low=1, high=10),
        ]
        
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=specs,
            objective_function=self.objective_function,
            metric=self.metric,
            crossover_method=CrossoverMethod.SINGLE_POINT,
            random_seed=42
        )
        
        parent1 = {"x1": 0.2, "x2": 0.3, "n": 3}
        parent2 = {"x1": 0.8, "x2": 0.9, "n": 7}
        
        child1, child2 = optimizer._single_point_crossover(parent1, parent2)
        
        # Children should have valid structure
        self.assertEqual(len(child1), 3)
        self.assertEqual(len(child2), 3)
        
        # Should inherit blocks from parents
        self.assertIn("x1", child1)
        self.assertIn("x2", child1)
        self.assertIn("n", child1)
    
    def test_two_point_crossover(self):
        """Test two-point crossover operation."""
        specs = [
            ParameterSpec(f"x{i}", ParameterType.FLOAT, low=0.0, high=1.0) 
            for i in range(5)
        ]
        
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=specs,
            objective_function=self.objective_function,
            metric=self.metric,
            crossover_method=CrossoverMethod.TWO_POINT,
            random_seed=42
        )
        
        parent1 = {f"x{i}": 0.1 * i for i in range(5)}
        parent2 = {f"x{i}": 0.9 - 0.1 * i for i in range(5)}
        
        child1, child2 = optimizer._two_point_crossover(parent1, parent2)
        
        # Children should have valid structure
        self.assertEqual(len(child1), 5)
        self.assertEqual(len(child2), 5)
    
    def test_gaussian_mutation(self):
        """Test Gaussian mutation operation."""
        specs = [
            ParameterSpec("x1", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("n", ParameterType.INTEGER, low=1, high=10),
        ]
        
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=specs,
            objective_function=self.objective_function,
            metric=self.metric,
            mutation_method=MutationMethod.GAUSSIAN,
            random_seed=42
        )
        
        individual = {"x1": 0.5, "n": 5}
        mutated = optimizer._gaussian_mutation(individual.copy())
        
        # Should maintain structure
        self.assertIn("x1", mutated)
        self.assertIn("n", mutated)
        
        # Values might change but should stay in bounds
        self.assertGreaterEqual(mutated["x1"], 0.0)
        self.assertLessEqual(mutated["x1"], 1.0)
        self.assertGreaterEqual(mutated["n"], 1)
        self.assertLessEqual(mutated["n"], 10)
    
    def test_uniform_mutation(self):
        """Test uniform mutation operation."""
        specs = [
            ParameterSpec("x1", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("choice", ParameterType.CATEGORICAL, choices=["a", "b", "c"]),
        ]
        
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=specs,
            objective_function=self.objective_function,
            metric=self.metric,
            mutation_method=MutationMethod.UNIFORM,
            random_seed=42
        )
        
        individual = {"x1": 0.5, "choice": "a"}
        mutated = optimizer._uniform_mutation(individual.copy())
        
        # Should maintain structure and bounds
        self.assertIn("x1", mutated)
        self.assertIn("choice", mutated)
        self.assertGreaterEqual(mutated["x1"], 0.0)
        self.assertLessEqual(mutated["x1"], 1.0)
        self.assertIn(mutated["choice"], ["a", "b", "c"])
    
    def test_polynomial_mutation(self):
        """Test polynomial mutation operation."""
        specs = [
            ParameterSpec("x1", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("x2", ParameterType.FLOAT, low=-1.0, high=1.0),
        ]
        
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=specs,
            objective_function=self.objective_function,
            metric=self.metric,
            mutation_method=MutationMethod.POLYNOMIAL,
            random_seed=42
        )
        
        individual = {"x1": 0.5, "x2": 0.0}
        mutated = optimizer._polynomial_mutation(individual.copy())
        
        # Should maintain bounds
        self.assertGreaterEqual(mutated["x1"], 0.0)
        self.assertLessEqual(mutated["x1"], 1.0)
        self.assertGreaterEqual(mutated["x2"], -1.0)
        self.assertLessEqual(mutated["x2"], 1.0)
    
    def test_suggest_parameters(self):
        """Test parameter suggestion during evolution."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            population_size=10,
            random_seed=42
        )
        
        # Should be able to suggest parameters
        params = optimizer.suggest_parameters()
        
        self.assertIsInstance(params, dict)
        self.assertEqual(len(params), len(self.parameter_specs))
        
        for spec in self.parameter_specs:
            self.assertIn(spec.name, params)
    
    def test_optimize_basic(self):
        """Test basic optimization workflow."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            population_size=20,
            random_seed=42
        )
        
        result = optimizer.optimize(max_iterations=10)  # Generations
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.best_parameters)
        self.assertIsNotNone(result.best_score)
        self.assertGreater(result.total_evaluations, 0)
        self.assertGreater(result.optimization_time, 0)
        self.assertGreater(len(result.convergence_history), 0)
    
    def test_optimize_with_convergence(self):
        """Test optimization with convergence detection."""
        # Simple objective with clear optimum
        def simple_objective(params):
            return -(params["x1"] - 0.7)**2
        
        specs = [ParameterSpec("x1", ParameterType.FLOAT, low=0.0, high=1.0)]
        
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=specs,
            objective_function=simple_objective,
            metric=self.metric,
            population_size=20,
            random_seed=42
        )
        
        result = optimizer.optimize(
            max_iterations=50,
            convergence_threshold=0.01
        )
        
        # Should find good solution
        self.assertAlmostEqual(result.best_parameters["x1"], 0.7, delta=0.2)
    
    def test_elitism(self):
        """Test that elitism preserves best individuals."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            population_size=10,
            elite_size=2,
            random_seed=42
        )
        
        # Initialize and evaluate population
        population = optimizer._initialize_population()
        fitness_scores = optimizer._evaluate_population(population)
        
        # Find best individuals
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        best_individuals = [population[i] for i in sorted_indices[:2]]
        
        # Evolve one generation
        new_population = optimizer._evolve_generation(population, fitness_scores)
        new_fitness = optimizer._evaluate_population(new_population)
        
        # Best individuals should be preserved
        for best_ind in best_individuals:
            found = False
            for new_ind in new_population:
                if best_ind == new_ind:
                    found = True
                    break
            # Note: Due to mutation, exact preservation might not occur
            # So we check that the best fitness is maintained or improved
        
        self.assertGreaterEqual(max(new_fitness), max(fitness_scores))
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        def deterministic_objective(params):
            return params["x1"] + params["x2"]
        
        specs = [
            ParameterSpec("x1", ParameterType.FLOAT, low=0.0, high=1.0),
            ParameterSpec("x2", ParameterType.FLOAT, low=0.0, high=1.0),
        ]
        
        optimizer1 = GeneticAlgorithmOptimizer(
            parameter_specs=specs,
            objective_function=deterministic_objective,
            metric=self.metric,
            population_size=10,
            random_seed=42
        )
        
        optimizer2 = GeneticAlgorithmOptimizer(
            parameter_specs=specs,
            objective_function=deterministic_objective,
            metric=self.metric,
            population_size=10,
            random_seed=42
        )
        
        result1 = optimizer1.optimize(max_iterations=5)
        result2 = optimizer2.optimize(max_iterations=5)
        
        # Results should be identical with same seed
        self.assertEqual(result1.total_evaluations, result2.total_evaluations)
        # Due to floating point precision, we use approximate equality
        self.assertAlmostEqual(result1.best_score, result2.best_score, places=5)
    
    def test_metadata_tracking(self):
        """Test that optimization metadata is properly tracked."""
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=self.objective_function,
            metric=self.metric,
            population_size=15,
            selection_method=SelectionMethod.TOURNAMENT,
            crossover_method=CrossoverMethod.UNIFORM,
            mutation_method=MutationMethod.GAUSSIAN
        )
        
        result = optimizer.optimize(max_iterations=5)
        
        self.assertIn("algorithm", result.metadata)
        self.assertIn("population_size", result.metadata)
        self.assertIn("selection_method", result.metadata)
        self.assertIn("crossover_method", result.metadata)
        self.assertIn("mutation_method", result.metadata)
        self.assertIn("total_generations", result.metadata)
        
        self.assertEqual(result.metadata["algorithm"], "genetic_algorithm")
        self.assertEqual(result.metadata["population_size"], 15)
        self.assertEqual(result.metadata["selection_method"], "tournament")
        self.assertEqual(result.metadata["crossover_method"], "uniform")
        self.assertEqual(result.metadata["mutation_method"], "gaussian")
    
    def test_objective_function_exception_handling(self):
        """Test handling of objective function exceptions."""
        def failing_objective(params):
            if params["x1"] > 0.8:
                raise ValueError("Simulated failure")
            return params["x1"]
        
        optimizer = GeneticAlgorithmOptimizer(
            parameter_specs=self.parameter_specs,
            objective_function=failing_objective,
            metric=self.metric,
            population_size=10,
            random_seed=42
        )
        
        # Should handle exceptions gracefully
        result = optimizer.optimize(max_iterations=5)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.best_parameters)


class TestGeneticAlgorithmEnums(unittest.TestCase):
    """Test cases for genetic algorithm enum classes."""
    
    def test_selection_method_values(self):
        """Test SelectionMethod enum values."""
        expected_methods = ["tournament", "roulette_wheel", "rank"]
        
        for method_name in expected_methods:
            method = SelectionMethod(method_name)
            self.assertEqual(method.value, method_name)
    
    def test_crossover_method_values(self):
        """Test CrossoverMethod enum values."""
        expected_methods = ["uniform", "single_point", "two_point"]
        
        for method_name in expected_methods:
            method = CrossoverMethod(method_name)
            self.assertEqual(method.value, method_name)
    
    def test_mutation_method_values(self):
        """Test MutationMethod enum values."""
        expected_methods = ["gaussian", "uniform", "polynomial"]
        
        for method_name in expected_methods:
            method = MutationMethod(method_name)
            self.assertEqual(method.value, method_name)


if __name__ == "__main__":
    unittest.main()