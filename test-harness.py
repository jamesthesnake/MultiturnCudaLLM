"""
Test Harness for comparing RL vs RL+Search performance
Evaluates the impact of test-time compute scaling
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time
import json
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm

from cuda_rl_system import BeamSearch, MCTS, KernelEvaluator
from gemm_specialist import GEMMSpecialist, create_gemm_specialist, NAIVE_GEMM, TILED_GEMM

@dataclass
class ExperimentResult:
    """Results from a single optimization experiment"""
    method: str  # "rl", "rl_beam", "rl_mcts"
    problem_id: str
    initial_gflops: float
    final_gflops: float
    speedup: float
    num_turns: int
    optimization_time: float
    trajectory_length: int
    optimizations_applied: List[str]
    search_candidates_evaluated: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)

class TestHarness:
    """Test harness for comparing different optimization approaches"""
    
    def __init__(self, 
                 kernelbench_path: Optional[str] = None,
                 output_dir: str = "results"):
        self.kernelbench_path = kernelbench_path
        self.output_dir = output_dir
        self.evaluator = KernelEvaluator(kernelbench_path)
        
        # Create specialists with different search algorithms
        self.specialists = {
            'rl': self._create_rl_specialist(),
            'rl_beam': self._create_beam_specialist(),
            'rl_mcts': self._create_mcts_specialist()
        }
        
        #
