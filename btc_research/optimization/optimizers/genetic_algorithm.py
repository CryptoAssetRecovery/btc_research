"""
Genetic Algorithm optimization for population-based evolutionary search.

This module implements a genetic algorithm with various selection, crossover,
and mutation operators for multi-objective optimization of trading strategies.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from btc_research.optimization.base import BaseOptimizer, ObjectiveFunction
from btc_research.optimization.types import (
    OptimizationMethod,
    OptimizationMetric,
    OptimizationResult,
    ParameterSpec,
    ParameterType,
)

__all__ = ["GeneticAlgorithmOptimizer", "SelectionMethod", "CrossoverMethod", "MutationMethod"]


class SelectionMethod:
    """Enumeration of supported selection methods."""
    
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    ELITIST = "elitist"


class CrossoverMethod:
    """Enumeration of supported crossover methods."""
    
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLEND_ALPHA = "blend_alpha"


class MutationMethod:
    """Enumeration of supported mutation methods."""
    
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"


class Individual:
    """Represents an individual in the genetic algorithm population."""
    
    def __init__(self, genes: np.ndarray, fitness: Optional[float] = None):
        """
        Initialize individual.
        
        Args:
            genes: Parameter vector (chromosome)
            fitness: Fitness value (objective function result)
        """
        self.genes = genes.copy()
        self.fitness = fitness
        self.age = 0
        self.rank = 0
    
    def __lt__(self, other):
        """Compare individuals by fitness (for sorting)."""
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness > other.fitness  # Assuming maximization
    
    def copy(self):
        """Create a copy of the individual."""
        new_individual = Individual(self.genes, self.fitness)
        new_individual.age = self.age
        new_individual.rank = self.rank
        return new_individual


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimization implementation.
    
    This optimizer uses evolutionary computation principles with population-based
    search, selection, crossover, and mutation operators to explore the parameter space.
    """
    
    def __init__(
        self,
        parameter_specs: List[ParameterSpec],
        objective_function: ObjectiveFunction,
        metric: OptimizationMetric,
        maximize: bool = True,
        random_seed: Optional[int] = None,
        population_size: int = 50,
        selection_method: str = SelectionMethod.TOURNAMENT,
        crossover_method: str = CrossoverMethod.UNIFORM,
        mutation_method: str = MutationMethod.GAUSSIAN,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.1,
        tournament_size: int = 3,
        mutation_strength: float = 0.1,
        adaptive_mutation: bool = True,
        multi_objective: bool = False,
        diversity_preservation: bool = True,
    ):
        """
        Initialize Genetic Algorithm optimizer.
        
        Args:
            parameter_specs: List of parameter specifications defining search space
            objective_function: Function to optimize (takes params dict, returns score)
            metric: Primary optimization metric
            maximize: Whether to maximize (True) or minimize (False) the objective
            random_seed: Random seed for reproducibility
            population_size: Number of individuals in population
            selection_method: Selection strategy for parent selection
            crossover_method: Crossover operator for recombination
            mutation_method: Mutation operator for diversity
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation per gene
            elitism_rate: Percentage of elite individuals to preserve
            tournament_size: Number of individuals in tournament selection
            mutation_strength: Standard deviation for Gaussian mutation
            adaptive_mutation: Whether to adapt mutation rate based on fitness diversity
            multi_objective: Whether to use multi-objective optimization (NSGA-II style)
            diversity_preservation: Whether to maintain population diversity
        """
        super().__init__(parameter_specs, objective_function, metric, maximize, random_seed)
        
        self.population_size = population_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.mutation_strength = mutation_strength
        self.adaptive_mutation = adaptive_mutation
        self.multi_objective = multi_objective
        self.diversity_preservation = diversity_preservation
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_seed)
        
        # Store the seed for potential reseeding
        self._random_seed = random_seed
        
        # Parameter space setup
        self._setup_parameter_space()
        
        # Population
        self.population: List[Individual] = []
        self.generation = 0
        
        # Statistics tracking
        self._fitness_history = []
        self._diversity_history = []
        self._best_individual = None
        
        # Adaptive parameters
        self._current_mutation_rate = mutation_rate
        self._stagnation_count = 0
        
        # State for suggest-evaluate-update cycle
        self._current_individual_idx = 0
        self._pending_evaluation = None
        self._generation_complete = False
        
        # Validation
        if population_size < 4:
            raise ValueError("Population size must be at least 4")
        
        if not (0 <= crossover_rate <= 1):
            raise ValueError("Crossover rate must be between 0 and 1")
        
        if not (0 <= mutation_rate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1")
    
    def _setup_parameter_space(self) -> None:
        """Setup parameter space bounds and transformations."""
        self.bounds = []
        self.param_names = []
        self.param_types = []
        self.categorical_mappings = {}
        
        for spec in self.parameter_specs:
            self.param_names.append(spec.name)
            self.param_types.append(spec.param_type)
            
            if spec.param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
                self.bounds.append((spec.low, spec.high))
            elif spec.param_type == ParameterType.CATEGORICAL:
                # Map categorical values to integers
                self.categorical_mappings[spec.name] = {
                    val: idx for idx, val in enumerate(spec.choices)
                }
                self.bounds.append((0, len(spec.choices) - 1))
            elif spec.param_type == ParameterType.BOOLEAN:
                self.bounds.append((0, 1))
        
        self.bounds = np.array(self.bounds)
        self.n_dims = len(self.bounds)
    
    def _encode_parameters(self, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Encode parameter dictionary to numerical vector.
        
        Args:
            parameters: Parameter dictionary
            
        Returns:
            Numerical vector representation
        """
        genes = np.zeros(self.n_dims)
        
        for i, (spec, param_name) in enumerate(zip(self.parameter_specs, self.param_names)):
            value = parameters[param_name]
            
            if spec.param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
                genes[i] = value
            elif spec.param_type == ParameterType.CATEGORICAL:
                genes[i] = self.categorical_mappings[param_name][value]
            elif spec.param_type == ParameterType.BOOLEAN:
                genes[i] = 1 if value else 0
        
        return genes
    
    def _decode_parameters(self, genes: np.ndarray) -> Dict[str, Any]:
        """
        Decode numerical vector to parameter dictionary.
        
        Args:
            genes: Numerical vector
            
        Returns:
            Parameter dictionary
        """
        parameters = {}
        
        for i, (spec, param_name) in enumerate(zip(self.parameter_specs, self.param_names)):
            value = genes[i]
            
            if spec.param_type == ParameterType.INTEGER:
                parameters[param_name] = int(np.round(np.clip(value, spec.low, spec.high)))
            elif spec.param_type == ParameterType.FLOAT:
                parameters[param_name] = float(np.clip(value, spec.low, spec.high))
            elif spec.param_type == ParameterType.CATEGORICAL:
                idx = int(np.round(np.clip(value, 0, len(spec.choices) - 1)))
                parameters[param_name] = spec.choices[idx]
            elif spec.param_type == ParameterType.BOOLEAN:
                parameters[param_name] = bool(np.round(value))
        
        return parameters
    
    def _create_random_individual(self) -> Individual:
        """Create a random individual within parameter bounds."""
        genes = np.zeros(self.n_dims)
        
        for i, (low, high) in enumerate(self.bounds):
            # Ensure we're generating truly random values
            genes[i] = self.rng.uniform(low, high)
        
        return Individual(genes)
    
    def _evaluate_individual(self, individual: Individual) -> float:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness value
        """
        if individual.fitness is not None:
            return individual.fitness
        
        try:
            parameters = self._decode_parameters(individual.genes)
            fitness = self.evaluate_parameters(parameters)
            individual.fitness = fitness
            return fitness
        except Exception:
            # Return worst possible fitness for failed evaluations
            individual.fitness = float('-inf') if self.maximize else float('inf')
            return individual.fitness
    
    def _initialize_population(self) -> None:
        """Initialize random population."""
        self.population = []
        for i in range(self.population_size):
            individual = self._create_random_individual()
            # Don't evaluate fitness here - let the suggest-evaluate-update cycle handle it
            self.population.append(individual)
        
        # Initialize best individual as None - will be set during evaluation
        self._best_individual = None
    
    def _tournament_selection(self, k: int = None) -> Individual:
        """
        Tournament selection of parent.
        
        Args:
            k: Tournament size (uses self.tournament_size if None)
            
        Returns:
            Selected individual
        """
        if k is None:
            k = self.tournament_size
        
        # Select k random individuals
        tournament = self.rng.choice(self.population, size=min(k, len(self.population)), replace=False)
        
        # Return best individual from tournament
        best = max(tournament, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
        return best
    
    def _roulette_wheel_selection(self) -> Individual:
        """
        Roulette wheel selection based on fitness proportions.
        
        Returns:
            Selected individual
        """
        fitness_values = [ind.fitness for ind in self.population if ind.fitness is not None]
        
        if not fitness_values:
            return self.rng.choice(self.population)
        
        # Adjust for minimization
        if not self.maximize:
            max_fitness = max(fitness_values)
            fitness_values = [max_fitness - f + 1 for f in fitness_values]
        
        # Handle negative fitness values
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1 for f in fitness_values]
        
        # Calculate probabilities
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            return self.rng.choice(self.population)
        
        probabilities = [f / total_fitness for f in fitness_values]
        
        # Select individual
        r = self.rng.random()
        cumulative_prob = 0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return self.population[i]
        
        return self.population[-1]
    
    def _rank_based_selection(self) -> Individual:
        """
        Rank-based selection.
        
        Returns:
            Selected individual
        """
        # Assign ranks (assuming population is sorted)
        n = len(self.population)
        ranks = list(range(n, 0, -1))  # Higher rank for better fitness
        
        total_rank = sum(ranks)
        probabilities = [rank / total_rank for rank in ranks]
        
        # Select individual
        r = self.rng.random()
        cumulative_prob = 0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return self.population[i]
        
        return self.population[-1]
    
    def _select_parent(self) -> Individual:
        """
        Select parent using configured selection method.
        
        Returns:
            Selected parent individual
        """
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection()
        elif self.selection_method == SelectionMethod.ROULETTE_WHEEL:
            return self._roulette_wheel_selection()
        elif self.selection_method == SelectionMethod.RANK_BASED:
            return self._rank_based_selection()
        else:
            # Default to tournament
            return self._tournament_selection()
    
    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Single-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        if self.n_dims <= 1:
            return parent1.copy(), parent2.copy()
        
        crossover_point = self.rng.randint(1, self.n_dims)
        
        child1_genes = np.concatenate([parent1.genes[:crossover_point], parent2.genes[crossover_point:]])
        child2_genes = np.concatenate([parent2.genes[:crossover_point], parent1.genes[crossover_point:]])
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Two-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        if self.n_dims <= 2:
            return self._single_point_crossover(parent1, parent2)
        
        point1, point2 = sorted(self.rng.choice(range(1, self.n_dims), size=2, replace=False))
        
        child1_genes = np.concatenate([
            parent1.genes[:point1],
            parent2.genes[point1:point2],
            parent1.genes[point2:]
        ])
        
        child2_genes = np.concatenate([
            parent2.genes[:point1],
            parent1.genes[point1:point2],
            parent2.genes[point2:]
        ])
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Uniform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        mask = self.rng.random(self.n_dims) < 0.5
        
        child1_genes = np.where(mask, parent1.genes, parent2.genes)
        child2_genes = np.where(mask, parent2.genes, parent1.genes)
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _arithmetic_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Arithmetic crossover (blending).
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        alpha = self.rng.random()
        
        child1_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
        child2_genes = alpha * parent2.genes + (1 - alpha) * parent1.genes
        
        # Ensure bounds are respected
        child1_genes = np.clip(child1_genes, self.bounds[:, 0], self.bounds[:, 1])
        child2_genes = np.clip(child2_genes, self.bounds[:, 0], self.bounds[:, 1])
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform crossover using configured method.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        if self.crossover_method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.ARITHMETIC:
            return self._arithmetic_crossover(parent1, parent2)
        else:
            # Default to uniform
            return self._uniform_crossover(parent1, parent2)
    
    def _gaussian_mutation(self, individual: Individual) -> Individual:
        """
        Gaussian mutation.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(self.n_dims):
            if self.rng.random() < self._current_mutation_rate:
                # Gaussian noise scaled by parameter range
                param_range = self.bounds[i, 1] - self.bounds[i, 0]
                noise = self.rng.normal(0, self.mutation_strength * param_range)
                mutated.genes[i] += noise
                
                # Ensure bounds
                mutated.genes[i] = np.clip(mutated.genes[i], self.bounds[i, 0], self.bounds[i, 1])
        
        mutated.fitness = None  # Reset fitness
        return mutated
    
    def _uniform_mutation(self, individual: Individual) -> Individual:
        """
        Uniform mutation.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(self.n_dims):
            if self.rng.random() < self._current_mutation_rate:
                # Uniform random value within bounds
                mutated.genes[i] = self.rng.uniform(self.bounds[i, 0], self.bounds[i, 1])
        
        mutated.fitness = None  # Reset fitness
        return mutated
    
    def _mutate(self, individual: Individual) -> Individual:
        """
        Perform mutation using configured method.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        if self.mutation_method == MutationMethod.GAUSSIAN:
            return self._gaussian_mutation(individual)
        elif self.mutation_method == MutationMethod.UNIFORM:
            return self._uniform_mutation(individual)
        else:
            # Default to Gaussian
            return self._gaussian_mutation(individual)
    
    def _calculate_diversity(self) -> float:
        """
        Calculate population diversity (average pairwise distance).
        
        Returns:
            Population diversity measure
        """
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Normalized Euclidean distance
                diff = self.population[i].genes - self.population[j].genes
                ranges = self.bounds[:, 1] - self.bounds[:, 0]
                normalized_diff = diff / ranges
                distance = np.sqrt(np.sum(normalized_diff ** 2))
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _update_adaptive_parameters(self) -> None:
        """Update adaptive parameters based on population state."""
        if not self.adaptive_mutation:
            return
        
        # Calculate fitness diversity
        fitness_values = [ind.fitness for ind in self.population if ind.fitness is not None]
        if len(fitness_values) < 2:
            return
        
        fitness_std = np.std(fitness_values)
        fitness_mean = np.mean(fitness_values)
        
        # Coefficient of variation
        cv = fitness_std / abs(fitness_mean) if fitness_mean != 0 else 0
        
        # Adapt mutation rate based on diversity
        if cv < 0.01:  # Low diversity
            self._current_mutation_rate = min(self.mutation_rate * 2, 0.5)
            self._stagnation_count += 1
        elif cv > 0.1:  # High diversity
            self._current_mutation_rate = max(self.mutation_rate * 0.5, 0.01)
            self._stagnation_count = 0
        else:
            self._current_mutation_rate = self.mutation_rate
            self._stagnation_count = 0
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """
        Suggest next parameter combination to evaluate.
        
        Returns:
            Dictionary of parameter names and suggested values
        """
        if not self.population:
            # Initialize population
            self._initialize_population()
            self._current_individual_idx = 0
            self._generation_complete = False
        
        # Check if we need to evolve to next generation
        if self._generation_complete:
            self._evolve_generation()
            self._current_individual_idx = 0
            self._generation_complete = False
        
        # Get current individual to evaluate
        if self._current_individual_idx < len(self.population):
            individual = self.population[self._current_individual_idx]
            self._pending_evaluation = individual
            parameters = self._decode_parameters(individual.genes)
            
            # Debug: print if parameters are identical
            if hasattr(self, '_last_suggested_params'):
                if parameters == self._last_suggested_params:
                    print(f"WARNING: Identical parameters suggested! Gen {self.generation}, Ind {self._current_individual_idx}")
            self._last_suggested_params = parameters.copy()
            
            return parameters
        else:
            # All individuals evaluated, mark generation as complete
            self._generation_complete = True
            return self.suggest_parameters()  # Recursive call to get next generation
    
    def update_with_result(self, parameters: Dict[str, Any], score: float) -> None:
        """
        Update optimizer state with evaluation result.
        
        Args:
            parameters: Parameter values that were evaluated
            score: Objective function result
        """
        super().update_with_result(parameters, score)
        
        # Update the pending evaluation
        if self._pending_evaluation is not None:
            self._pending_evaluation.fitness = score
            
            # Update best individual if this is better
            if (self._best_individual is None or 
                (self.maximize and score > self._best_individual.fitness) or
                (not self.maximize and score < self._best_individual.fitness)):
                self._best_individual = self._pending_evaluation.copy()
            
            # Move to next individual
            self._current_individual_idx += 1
            self._pending_evaluation = None
            
            # Check if all individuals in current generation have been evaluated
            if self._current_individual_idx >= len(self.population):
                self._generation_complete = True
        else:
            # Fallback: find corresponding individual in population
            genes = self._encode_parameters(parameters)
            for individual in self.population:
                if np.allclose(individual.genes, genes, atol=1e-6):
                    individual.fitness = score
                    break
    
    def _evolve_generation(self) -> None:
        """Evolve population by one generation."""
        # Ensure all individuals have been evaluated
        evaluated_population = [ind for ind in self.population if ind.fitness is not None]
        
        if len(evaluated_population) == 0:
            # No individuals evaluated yet, can't evolve
            return
        
        # Sort population by fitness (only evaluated individuals)
        evaluated_population.sort(key=lambda x: x.fitness, reverse=self.maximize)
        
        # Update best individual
        if (self._best_individual is None or 
            (self.maximize and evaluated_population[0].fitness > self._best_individual.fitness) or
            (not self.maximize and evaluated_population[0].fitness < self._best_individual.fitness)):
            self._best_individual = evaluated_population[0].copy()
        
        # Calculate statistics
        diversity = self._calculate_diversity()
        fitness_values = [ind.fitness for ind in evaluated_population]
        
        self._diversity_history.append(diversity)
        if fitness_values:
            self._fitness_history.append({
                'best': max(fitness_values) if self.maximize else min(fitness_values),
                'mean': np.mean(fitness_values),
                'std': np.std(fitness_values)
            })
        
        # Update adaptive parameters
        self._update_adaptive_parameters()
        
        # Update population to use evaluated individuals
        self.population = evaluated_population
        
        # Create new population
        new_population = []
        
        # Elitism: preserve best individuals
        n_elite = int(self.elitism_rate * self.population_size)
        new_population.extend([ind.copy() for ind in self.population[:n_elite]])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if self.rng.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            # Don't evaluate fitness here - let the suggest-evaluate-update cycle handle it
            # Reset fitness to indicate they need evaluation
            child1.fitness = None
            child2.fitness = None
            
            # Add to new population
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        new_population = new_population[:self.population_size]
        
        # Update population
        self.population = new_population
        self.generation += 1
        
        # Age tracking
        for individual in self.population:
            individual.age += 1
    
    def optimize(
        self,
        max_iterations: int = 100,
        timeout_seconds: Optional[float] = None,
        convergence_threshold: Optional[float] = None,
        max_generations: Optional[int] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Run genetic algorithm optimization.
        
        Args:
            max_iterations: Maximum number of function evaluations
            timeout_seconds: Maximum time to run optimization
            convergence_threshold: Stop if improvement falls below this threshold
            max_generations: Maximum number of generations (overrides max_iterations)
            **kwargs: Additional parameters
            
        Returns:
            Best optimization result found
        """
        start_time = datetime.now()
        
        # Reset state
        self.population.clear()
        self.generation = 0
        self._fitness_history.clear()
        self._diversity_history.clear()
        self._best_individual = None
        self._current_mutation_rate = self.mutation_rate
        self._stagnation_count = 0
        
        # Reset suggest-evaluate-update state
        self._current_individual_idx = 0
        self._pending_evaluation = None
        self._generation_complete = False
        
        # Initialize population
        self._initialize_population()
        
        # Determine stopping criteria
        if max_generations is not None:
            max_gens = max_generations
        else:
            max_gens = max_iterations // self.population_size
        
        # Evolution loop
        for generation in range(max_gens):
            # Check timeout
            if timeout_seconds:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    print(f"Genetic algorithm stopped due to timeout after {generation} generations")
                    break
            
            # Check convergence
            if convergence_threshold is not None and len(self._fitness_history) >= 2:
                recent_improvement = abs(
                    self._fitness_history[-1]['best'] - self._fitness_history[-2]['best']
                )
                if recent_improvement < convergence_threshold:
                    print(f"Genetic algorithm converged after {generation} generations")
                    break
            
            # Check stagnation
            if self._stagnation_count > 10:  # No improvement for 10 generations
                print(f"Genetic algorithm stopped due to stagnation after {generation} generations")
                break
            
            try:
                # Evolve one generation
                self._evolve_generation()
                
            except Exception as e:
                print(f"Warning: Generation {generation} failed: {e}")
                continue
        
        end_time = datetime.now()
        
        if self._best_individual is None:
            raise ValueError("No valid solutions found")
        
        # Create optimization result
        best_params = self._decode_parameters(self._best_individual.genes)
        
        result = OptimizationResult(
            parameters=best_params,
            metrics={"objective_value": self._best_individual.fitness},
            objective_value=self._best_individual.fitness,
            in_sample_metrics={"objective_value": self._best_individual.fitness},
            method=OptimizationMethod.GENETIC_ALGORITHM,
            metric=self.metric,
            start_time=start_time,
            end_time=end_time,
            iterations=self.generation * self.population_size,
            convergence_achieved=self._stagnation_count <= 10,
            diagnostics={
                "generations": self.generation,
                "population_size": self.population_size,
                "selection_method": self.selection_method,
                "crossover_method": self.crossover_method,
                "mutation_method": self.mutation_method,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "final_mutation_rate": self._current_mutation_rate,
                "elitism_rate": self.elitism_rate,
                "stagnation_count": self._stagnation_count,
                "final_diversity": self._diversity_history[-1] if self._diversity_history else 0,
                "fitness_history": self._fitness_history,
                "diversity_history": self._diversity_history,
            }
        )
        
        self._best_result = result
        return result
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """
        Get current population statistics.
        
        Returns:
            Dictionary with population statistics
        """
        if not self.population:
            return {}
        
        fitness_values = [ind.fitness for ind in self.population if ind.fitness is not None]
        
        if not fitness_values:
            return {"generation": self.generation, "population_size": len(self.population)}
        
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": max(fitness_values) if self.maximize else min(fitness_values),
            "mean_fitness": np.mean(fitness_values),
            "std_fitness": np.std(fitness_values),
            "diversity": self._calculate_diversity(),
            "mutation_rate": self._current_mutation_rate,
            "stagnation_count": self._stagnation_count,
        }
    
    def get_fitness_evolution(self) -> Dict[str, List[float]]:
        """
        Get fitness evolution over generations.
        
        Returns:
            Dictionary with fitness statistics over time
        """
        return {
            "best": [gen['best'] for gen in self._fitness_history],
            "mean": [gen['mean'] for gen in self._fitness_history],
            "std": [gen['std'] for gen in self._fitness_history],
            "diversity": self._diversity_history,
        }