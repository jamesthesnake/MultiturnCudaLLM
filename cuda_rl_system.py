"""
Multi-Turn RL System for CUDA Kernel Optimization
Inspired by ether0 and Kevin-32B approaches
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import subprocess
import tempfile
import re
import json
from abc import ABC, abstractmethod
import random
from enum import Enum

# ===== Data Structures =====

@dataclass
class KernelState:
    """Represents the state of a CUDA kernel at a given optimization turn"""
    code: str
    turn: int
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    optimization_applied: str = ""
    
@dataclass
class OptimizationTrajectory:
    """Represents a multi-turn optimization trajectory"""
    states: List[KernelState]
    total_reward: float = 0.0
    
    def add_state(self, state: KernelState):
        self.states.append(state)
        
    @property
    def current_state(self) -> KernelState:
        return self.states[-1] if self.states else None
    
    @property
    def num_turns(self) -> int:
        return len(self.states)

class OptimizationType(Enum):
    """Types of optimizations that can be applied"""
    MEMORY_COALESCING = "memory_coalescing"
    SHARED_MEMORY = "shared_memory"
    BANK_CONFLICT = "bank_conflict"
    THREAD_CONFIG = "thread_config"
    REGISTER_BLOCKING = "register_blocking"
    TENSOR_CORES = "tensor_cores"

# ===== Curriculum Buffer =====

class CurriculumBuffer:
    """
    Advantage-based curriculum buffer inspired by ether0
    Stores problems that show non-trivial variance in rewards
    """
    def __init__(self, capacity: int = 10000, variance_threshold: float = 0.1):
        self.buffer = deque(maxlen=capacity)
        self.variance_threshold = variance_threshold
        self.problem_stats = {}  # Track reward variance per problem
        
    def add_group(self, problem_id: str, rewards: List[float]):
        """Add a group of rewards for a problem"""
        if len(rewards) > 1:
            variance = np.var(rewards)
            if variance > self.variance_threshold:
                self.buffer.append(problem_id)
                self.problem_stats[problem_id] = {
                    'variance': variance,
                    'mean_reward': np.mean(rewards),
                    'num_samples': len(rewards)
                }
    
    def sample(self, batch_size: int, epsilon_cur: float = 0.5) -> List[str]:
        """Sample problems with epsilon_cur fraction from curriculum buffer"""
        curriculum_size = int(batch_size * epsilon_cur)
        dataset_size = batch_size - curriculum_size
        
        curriculum_samples = []
        if len(self.buffer) > 0:
            curriculum_samples = random.choices(
                list(self.buffer), 
                k=min(curriculum_size, len(self.buffer))
            )
        
        # Return curriculum samples (actual dataset sampling would happen elsewhere)
        return curriculum_samples
    
    def remove_trivial(self, problem_id: str):
        """Remove problems that have become trivial (low variance)"""
        if problem_id in self.problem_stats:
            del self.problem_stats[problem_id]
            # deque doesn't have efficient removal, so we rebuild
            self.buffer = deque(
                [p for p in self.buffer if p != problem_id], 
                maxlen=self.buffer.maxlen
            )

# ===== Kernel Evaluation =====

class KernelEvaluator:
    """Evaluates CUDA kernels using KernelBench or direct compilation"""
    
    def __init__(self, kernelbench_path: Optional[str] = None):
        self.kernelbench_path = kernelbench_path
        
    def evaluate(self, kernel_code: str, kernel_type: str = "gemm") -> Dict[str, Any]:
        """
        Evaluate a CUDA kernel and return performance metrics
        Returns: {
            'compiles': bool,
            'correct': bool,
            'gflops': float,
            'bandwidth_utilization': float,
            'occupancy': float,
            'error_msg': str (if compilation fails)
        }
        """
        # Write kernel to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(kernel_code)
            kernel_path = f.name
        
        try:
            # Try to compile
            compile_result = subprocess.run(
                ['nvcc', '-o', '/tmp/kernel_test', kernel_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if compile_result.returncode != 0:
                return {
                    'compiles': False,
                    'correct': False,
                    'gflops': 0.0,
                    'bandwidth_utilization': 0.0,
                    'occupancy': 0.0,
                    'error_msg': compile_result.stderr
                }
            
            # If using KernelBench
            if self.kernelbench_path and kernel_type == "gemm":
                # Run through KernelBench for detailed metrics
                # This is a placeholder - actual integration would depend on KernelBench API
                metrics = self._run_kernelbench(kernel_code, kernel_type)
            else:
                # Basic evaluation
                metrics = self._basic_evaluation(kernel_path)
            
            return metrics
            
        except Exception as e:
            return {
                'compiles': False,
                'correct': False,
                'gflops': 0.0,
                'bandwidth_utilization': 0.0,
                'occupancy': 0.0,
                'error_msg': str(e)
            }
        finally:
            # Cleanup
            import os
            if os.path.exists(kernel_path):
                os.remove(kernel_path)
    
    def _run_kernelbench(self, kernel_code: str, kernel_type: str) -> Dict[str, Any]:
        """Run kernel through KernelBench for evaluation"""
        # Placeholder for KernelBench integration
        # In practice, this would call KernelBench's evaluation API
        return {
            'compiles': True,
            'correct': True,
            'gflops': np.random.uniform(100, 500),  # Placeholder
            'bandwidth_utilization': np.random.uniform(0.3, 0.9),
            'occupancy': np.random.uniform(0.4, 0.8),
            'error_msg': ''
        }
    
    def _basic_evaluation(self, kernel_path: str) -> Dict[str, Any]:
        """Basic evaluation without KernelBench"""
        # Run the compiled kernel with test data
        # This is simplified - real implementation would set up proper test data
        try:
            run_result = subprocess.run(
                ['/tmp/kernel_test'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if run_result.returncode != 0:
                return {
                    'compiles': True,
                    'correct': False,
                    'gflops': 0.0,
                    'bandwidth_utilization': 0.0,
                    'occupancy': 0.0,
                    'error_msg': run_result.stderr
                }
            
            # Parse output for performance metrics (simplified)
            # Real implementation would parse actual performance counters
            return {
                'compiles': True,
                'correct': True,
                'gflops': np.random.uniform(50, 200),  # Placeholder
                'bandwidth_utilization': np.random.uniform(0.2, 0.6),
                'occupancy': np.random.uniform(0.3, 0.6),
                'error_msg': ''
            }
            
        except Exception as e:
            return {
                'compiles': True,
                'correct': False,
                'gflops': 0.0,
                'bandwidth_utilization': 0.0,
                'occupancy': 0.0,
                'error_msg': str(e)
            }

# ===== Reward Functions =====

class RewardCalculator:
    """Calculate rewards for kernel optimizations"""
    
    def __init__(self, turn_bonuses: Optional[Dict[int, Dict[str, float]]] = None):
        # Turn-specific bonuses for progressive optimization
        self.turn_bonuses = turn_bonuses or {
            1: {'shared_memory': 0.2, 'coalescing': 0.1},
            2: {'bank_conflict_fix': 0.3, 'occupancy_improvement': 0.2},
            3: {'tensor_cores': 0.4, 'register_blocking': 0.2}
        }
    
    def calculate_reward(self, 
                        current_metrics: Dict[str, Any],
                        previous_metrics: Optional[Dict[str, Any]],
                        turn: int,
                        optimization_type: str) -> float:
        """
        Calculate reward using product of format and accuracy rewards (like ether0)
        Now with real performance metrics!
        """
        # Format reward (does it compile?)
        format_reward = 1.0 if current_metrics['compiles'] else 0.0
        
        # Accuracy reward (is it correct?)
        accuracy_reward = 1.0 if current_metrics.get('correct', True) else 0.0
        
        # Performance reward based on real metrics
        perf_reward = 0.0
        
        if format_reward > 0 and accuracy_reward > 0:
            # Use actual GFLOPS if available
            current_gflops = current_metrics.get('gflops', 0)
            
            if previous_metrics and previous_metrics.get('gflops', 0) > 0:
                # Calculate speedup
                speedup = current_gflops / previous_metrics.get('gflops', 1)
                # Reward improvements, penalize regressions
                perf_reward = np.log2(speedup) if speedup > 0 else -1.0
                perf_reward = np.clip(perf_reward, -1, 2)  # Clip to reasonable range
            else:
                # First turn - reward based on efficiency
                # A40 peak is 37.4 TFLOPS
                efficiency = current_metrics.get('percent_of_peak', 0) / 100.0
                perf_reward = efficiency  # 0 to 1 based on peak utilization
            
            # Bonus for power efficiency
            if 'gflops_per_watt' in current_metrics:
                power_bonus = current_metrics['gflops_per_watt'] / 200.0  # Normalize by typical value
                perf_reward += 0.1 * np.clip(power_bonus, 0, 1)
            
            # Bonus for good occupancy
            occupancy = current_metrics.get('occupancy', 0.5)
            if occupancy > 0.7:
                perf_reward += 0.1
        
        # Turn-specific bonuses
        turn_bonus = 0.0
        if turn in self.turn_bonuses and optimization_type in self.turn_bonuses[turn]:
            turn_bonus = self.turn_bonuses[turn][optimization_type]
        
        # Product reward (like ether0) with performance component
        base_reward = format_reward * accuracy_reward * (0.3 + 0.7 * np.clip(perf_reward, 0, 1))
        
        # Add turn bonus
        total_reward = base_reward * (1 + turn_bonus)
        
        # Log reward components for debugging
        if current_gflops > 0:
            print(f"[RewardCalculator] Reward breakdown:")
            print(f"  Format: {format_reward}, Accuracy: {accuracy_reward}")
            print(f"  Performance: {perf_reward:.3f} (GFLOPS: {current_gflops:.1f})")
            print(f"  Turn bonus: {turn_bonus}")
            print(f"  Total reward: {total_reward:.3f}")
        
        return total_reward['compiles'] else 0.0
        
        # Accuracy reward (is it correct?)
        accuracy_reward = 1.0 if current_metrics['correct'] else 0.0
        
        # Performance reward (normalized improvement)
        perf_reward = 0.0
        if previous_metrics and current_metrics['gflops'] > 0:
            improvement = (current_metrics['gflops'] - previous_metrics.get('gflops', 0)) / max(previous_metrics.get('gflops', 1), 1)
            perf_reward = np.clip(improvement, -1, 2)  # Allow for some regression but cap gains
        elif current_metrics['gflops'] > 0:
            # First turn - normalize against a baseline
            perf_reward = current_metrics['gflops'] / 100.0  # Assume 100 GFLOPS as baseline
        
        # Turn-specific bonuses
        turn_bonus = 0.0
        if turn in self.turn_bonuses and optimization_type in self.turn_bonuses[turn]:
            turn_bonus = self.turn_bonuses[turn][optimization_type]
        
        # Product reward (like ether0) with performance component
        base_reward = format_reward * accuracy_reward * (0.5 + 0.5 * perf_reward)
        
        # Add turn bonus
        total_reward = base_reward + turn_bonus * base_reward
        
        return total_reward

# ===== Search Algorithms =====

class SearchAlgorithm(ABC):
    """Base class for search algorithms"""
    
    @abstractmethod
    def search(self, 
               initial_state: KernelState,
               generator,
               evaluator: KernelEvaluator,
               num_candidates: int,
               max_depth: int) -> List[OptimizationTrajectory]:
        pass

class BeamSearch(SearchAlgorithm):
    """Beam search for exploring optimization trajectories"""
    
    def __init__(self, beam_width: int = 8, reward_calculator: Optional[RewardCalculator] = None):
        self.beam_width = beam_width
        self.reward_calculator = reward_calculator or RewardCalculator()
    
    def search(self,
               initial_state: KernelState,
               generator,
               evaluator: KernelEvaluator,
               num_candidates: int = 16,
               max_depth: int = 3) -> List[OptimizationTrajectory]:
        """
        Perform beam search over optimization trajectories
        """
        # Initialize beam with single trajectory
        beam = [OptimizationTrajectory(states=[initial_state])]
        
        for turn in range(1, max_depth + 1):
            new_beam = []
            
            for trajectory in beam:
                current_state = trajectory.current_state
                
                # Generate candidates for this state
                candidates = generator.generate_optimizations(
                    current_state,
                    num_candidates=num_candidates,
                    turn=turn
                )
                
                for candidate_code, reasoning, optimization_type in candidates:
                    # Evaluate candidate
                    metrics = evaluator.evaluate(candidate_code)
                    
                    # Calculate reward
                    reward = self.reward_calculator.calculate_reward(
                        metrics,
                        current_state.performance_metrics,
                        turn,
                        optimization_type
                    )
                    
                    # Create new state
                    new_state = KernelState(
                        code=candidate_code,
                        turn=turn,
                        performance_metrics=metrics,
                        reasoning=reasoning,
                        optimization_applied=optimization_type
                    )
                    
                    # Create new trajectory
                    new_trajectory = OptimizationTrajectory(
                        states=trajectory.states + [new_state],
                        total_reward=trajectory.total_reward + reward
                    )
                    
                    new_beam.append(new_trajectory)
            
            # Keep top beam_width trajectories
            new_beam.sort(key=lambda t: t.total_reward, reverse=True)
            beam = new_beam[:self.beam_width]
        
        return beam

class MCTSNode:
    """Node for Monte Carlo Tree Search"""
    
    def __init__(self, state: KernelState, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = None
    
    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0
    
    def uct_value(self, c=1.414) -> float:
        """Upper Confidence Bound for Trees"""
        if self.visits == 0:
            return float('inf')
        return self.avg_reward + c * np.sqrt(np.log(self.parent.visits) / self.visits)

class MCTS(SearchAlgorithm):
    """Monte Carlo Tree Search for optimization exploration"""
    
    def __init__(self, 
                 num_simulations: int = 100,
                 c_param: float = 1.414,
                 reward_calculator: Optional[RewardCalculator] = None):
        self.num_simulations = num_simulations
        self.c_param = c_param
        self.reward_calculator = reward_calculator or RewardCalculator()
    
    def search(self,
               initial_state: KernelState,
               generator,
               evaluator: KernelEvaluator,
               num_candidates: int = 8,
               max_depth: int = 3) -> List[OptimizationTrajectory]:
        """
        Perform MCTS to find best optimization trajectories
        """
        root = MCTSNode(initial_state)
        
        for _ in range(self.num_simulations):
            node = self._select(root)
            reward = self._simulate(node, generator, evaluator, max_depth)
            self._backpropagate(node, reward)
        
        # Extract best trajectories
        trajectories = self._extract_trajectories(root, max_depth)
        return trajectories
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node to expand using UCT"""
        while node.children:
            node = max(node.children, key=lambda n: n.uct_value(self.c_param))
        return node
    
    def _simulate(self, node: MCTSNode, generator, evaluator, max_depth: int) -> float:
        """Simulate a random playout from the node"""
        current_state = node.state
        total_reward = 0.0
        
        for turn in range(current_state.turn + 1, max_depth + 1):
            # Generate random optimization
            candidates = generator.generate_optimizations(
                current_state,
                num_candidates=1,
                turn=turn
            )
            
            if not candidates:
                break
                
            candidate_code, reasoning, optimization_type = candidates[0]
            metrics = evaluator.evaluate(candidate_code)
            
            reward = self.reward_calculator.calculate_reward(
                metrics,
                current_state.performance_metrics,
                turn,
                optimization_type
            )
            
            total_reward += reward
            
            # Update current state for next iteration
            current_state = KernelState(
                code=candidate_code,
                turn=turn,
                performance_metrics=metrics,
                reasoning=reasoning,
                optimization_applied=optimization_type
            )
        
        return total_reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree"""
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def _extract_trajectories(self, root: MCTSNode, max_depth: int) -> List[OptimizationTrajectory]:
        """Extract best trajectories from MCTS tree"""
        trajectories = []
        
        def traverse(node, path):
            if node.state.turn >= max_depth or not node.children:
                # Leaf node - create trajectory
                trajectory = OptimizationTrajectory(states=path)
                trajectory.total_reward = sum(s.performance_metrics.get('gflops', 0) for s in path)
                trajectories.append(trajectory)
            else:
                # Continue traversal on best children
                sorted_children = sorted(node.children, key=lambda n: n.avg_reward, reverse=True)
                for child in sorted_children[:3]:  # Top 3 children
                    traverse(child, path + [child.state])
        
        traverse(root, [root.state])
        return sorted(trajectories, key=lambda t: t.total_reward, reverse=True)

# ===== Main Specialist Base Class =====

class CUDAOptimizationSpecialist(ABC):
    """Base class for CUDA optimization specialists"""
    
    def __init__(self, 
                 model_name: str,
                 optimization_types: List[OptimizationType],
                 evaluator: KernelEvaluator,
                 search_algorithm: SearchAlgorithm,
                 reward_calculator: RewardCalculator):
        self.model_name = model_name
        self.optimization_types = optimization_types
        self.evaluator = evaluator
        self.search_algorithm = search_algorithm
        self.reward_calculator = reward_calculator
        self.curriculum_buffer = CurriculumBuffer()
    
    @abstractmethod
    def generate_optimizations(self,
                             state: KernelState,
                             num_candidates: int,
                             turn: int) -> List[Tuple[str, str, str]]:
        """
        Generate optimization candidates
        Returns: List of (code, reasoning, optimization_type) tuples
        """
        pass
    
    def optimize_kernel(self,
                       initial_code: str,
                       problem_id: str,
                       use_search: bool = True,
                       max_turns: int = 3) -> OptimizationTrajectory:
        """
        Optimize a kernel using multi-turn RL with optional search
        """
        # Evaluate initial kernel
        initial_metrics = self.evaluator.evaluate(initial_code)
        initial_state = KernelState(
            code=initial_code,
            turn=0,
            performance_metrics=initial_metrics,
            reasoning="Initial kernel",
            optimization_applied="none"
        )
        
        if use_search:
            # Use search algorithm to find best trajectory
            trajectories = self.search_algorithm.search(
                initial_state,
                self,
                self.evaluator,
                num_candidates=16,
                max_depth=max_turns
            )
            
            # Update curriculum buffer with reward variance
            rewards = [t.total_reward for t in trajectories]
            self.curriculum_buffer.add_group(problem_id, rewards)
            
            return trajectories[0] if trajectories else OptimizationTrajectory(states=[initial_state])
        else:
            # Simple greedy optimization without search
            trajectory = OptimizationTrajectory(states=[initial_state])
            current_state = initial_state
            
            for turn in range(1, max_turns + 1):
                candidates = self.generate_optimizations(current_state, num_candidates=1, turn=turn)
                if not candidates:
                    break
                    
                code, reasoning, opt_type = candidates[0]
                metrics = self.evaluator.evaluate(code)
                
                reward = self.reward_calculator.calculate_reward(
                    metrics,
                    current_state.performance_metrics,
                    turn,
                    opt_type
                )
                
                new_state = KernelState(
                    code=code,
                    turn=turn,
                    performance_metrics=metrics,
                    reasoning=reasoning,
                    optimization_applied=opt_type
                )
                
                trajectory.add_state(new_state)
                trajectory.total_reward += reward
                current_state = new_state
            
            return trajectory
