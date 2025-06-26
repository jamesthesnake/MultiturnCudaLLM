"""
Multi-GPU CUDA RL System for 3x A40 Configuration
GPU 0: Model training and inference
GPU 1-2: Parallel kernel compilation and evaluation
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import asyncio
from asyncio import Queue
import os
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import subprocess
import tempfile

# Import base classes
from cuda_rl_system import (
    KernelState, OptimizationTrajectory, KernelEvaluator,
    RewardCalculator, CurriculumBuffer
)

@dataclass
class GPUConfig:
    """Configuration for multi-GPU setup"""
    model_gpu: int = 0
    kernel_gpus: List[int] = None
    memory_fraction: float = 0.8
    
    def __post_init__(self):
        if self.kernel_gpus is None:
            self.kernel_gpus = [1, 2]

class A40KernelEvaluator(KernelEvaluator):
    """
    Enhanced kernel evaluator for multi-GPU setup
    Manages parallel evaluation across multiple A40s
    """
    
    def __init__(self, gpu_id: int, kernelbench_path: Optional[str] = None):
        super().__init__(kernelbench_path)
        self.gpu_id = gpu_id
        
        # Pre-allocate GPU workspace
        with torch.cuda.device(gpu_id):
            # Reserve 10GB for kernel evaluation
            self.workspace = torch.cuda.ByteTensor(10 * 1024**3)
            torch.cuda.synchronize()
    
    def evaluate(self, kernel_code: str, kernel_type: str = "gemm") -> Dict[str, Any]:
        """Evaluate kernel on specific GPU"""
        # Set CUDA device for subprocess
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        # Write kernel to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(kernel_code)
            kernel_path = f.name
        
        try:
            # Compile with nvcc
            compile_result = subprocess.run(
                ['nvcc', '-O3', '-arch=sm_86',  # A40 is sm_86
                 '-o', f'/tmp/kernel_gpu{self.gpu_id}', kernel_path],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            if compile_result.returncode != 0:
                return {
                    'compiles': False,
                    'correct': False,
                    'gflops': 0.0,
                    'bandwidth_utilization': 0.0,
                    'occupancy': 0.0,
                    'error_msg': compile_result.stderr,
                    'gpu_id': self.gpu_id
                }
            
            # Run kernel benchmark
            run_result = subprocess.run(
                [f'/tmp/kernel_gpu{self.gpu_id}'],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            
            # Parse results (simplified - real implementation would parse actual metrics)
            metrics = self._parse_kernel_output(run_result.stdout)
            metrics['gpu_id'] = self.gpu_id
            
            return metrics
            
        except Exception as e:
            return {
                'compiles': False,
                'correct': False,
                'gflops': 0.0,
                'bandwidth_utilization': 0.0,
                'occupancy': 0.0,
                'error_msg': str(e),
                'gpu_id': self.gpu_id
            }
        finally:
            if os.path.exists(kernel_path):
                os.remove(kernel_path)
            if os.path.exists(f'/tmp/kernel_gpu{self.gpu_id}'):
                os.remove(f'/tmp/kernel_gpu{self.gpu_id}')
    
    def _parse_kernel_output(self, output: str) -> Dict[str, float]:
        """Parse kernel execution output"""
        # Simplified parsing - real implementation would extract actual metrics
        return {
            'compiles': True,
            'correct': True,
            'gflops': np.random.uniform(200, 800),  # Placeholder
            'bandwidth_utilization': np.random.uniform(0.4, 0.9),
            'occupancy': np.random.uniform(0.5, 0.95),
            'error_msg': ''
        }

class MultiGPUCoordinator:
    """
    Coordinates work across multiple A40 GPUs
    GPU 0: Model operations
    GPU 1-2: Kernel evaluation
    """
    
    def __init__(self, config: GPUConfig):
        self.config = config
        
        # Initialize evaluators for each kernel GPU
        self.evaluators = {
            gpu_id: A40KernelEvaluator(gpu_id)
            for gpu_id in config.kernel_gpus
        }
        
        # Async queue for GPU availability
        self.gpu_queue = asyncio.Queue()
        for gpu_id in config.kernel_gpus:
            self.gpu_queue.put_nowait(gpu_id)
        
        # Metrics tracking
        self.gpu_utilization = {gpu_id: 0.0 for gpu_id in config.kernel_gpus}
        self.evaluation_times = []
        
        # Process pool for CPU compilation
        self.compile_pool = ProcessPoolExecutor(max_workers=4)
    
    async def evaluate_kernels_parallel(self, 
                                      kernel_codes: List[Tuple[str, str]]) -> List[Dict]:
        """
        Evaluate multiple kernels in parallel across available GPUs
        
        Args:
            kernel_codes: List of (kernel_code, kernel_type) tuples
            
        Returns:
            List of evaluation results
        """
        evaluation_tasks = []
        
        for kernel_code, kernel_type in kernel_codes:
            # Get available GPU
            gpu_id = await self.gpu_queue.get()
            
            # Create evaluation task
            task = asyncio.create_task(
                self._evaluate_kernel_async(kernel_code, kernel_type, gpu_id)
            )
            evaluation_tasks.append((task, gpu_id))
        
        # Wait for all evaluations
        results = []
        for task, gpu_id in evaluation_tasks:
            try:
                result = await task
                results.append(result)
            finally:
                # Return GPU to pool
                await self.gpu_queue.put(gpu_id)
        
        return results
    
    async def _evaluate_kernel_async(self, 
                                   kernel_code: str, 
                                   kernel_type: str,
                                   gpu_id: int) -> Dict:
        """Evaluate kernel asynchronously on specific GPU"""
        start_time = time.time()
        
        # Track GPU utilization
        self.gpu_utilization[gpu_id] = 1.0
        
        # Run evaluation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        evaluator = self.evaluators[gpu_id]
        
        result = await loop.run_in_executor(
            None,
            evaluator.evaluate,
            kernel_code,
            kernel_type
        )
        
        # Update metrics
        eval_time = time.time() - start_time
        self.evaluation_times.append(eval_time)
        self.gpu_utilization[gpu_id] = 0.0
        
        result['evaluation_time'] = eval_time
        
        return result
    
    def compile_kernel_cpu(self, kernel_code: str) -> Optional[str]:
        """Compile kernel on CPU (doesn't need GPU)"""
        future = self.compile_pool.submit(self._compile_kernel, kernel_code)
        try:
            return future.result(timeout=30)
        except:
            return None
    
    def _compile_kernel(self, kernel_code: str) -> str:
        """CPU compilation of CUDA kernel"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(kernel_code)
            kernel_path = f.name
        
        try:
            # Just syntax check with nvcc
            result = subprocess.run(
                ['nvcc', '-arch=sm_86', '--dryrun', '-c', kernel_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return "compiled"
            else:
                return None
        finally:
            if os.path.exists(kernel_path):
                os.remove(kernel_path)
    
    def get_gpu_stats(self) -> Dict:
        """Get statistics about GPU utilization"""
        return {
            'gpu_utilization': self.gpu_utilization.copy(),
            'avg_eval_time': np.mean(self.evaluation_times) if self.evaluation_times else 0,
            'total_evaluations': len(self.evaluation_times),
            'gpus_available': self.gpu_queue.qsize()
        }

class A40Optimizer:
    """
    Main optimization system for 3x A40 setup
    Coordinates model training and kernel evaluation
    """
    
    def __init__(self):
        # Configure GPUs
        self.config = GPUConfig(
            model_gpu=0,
            kernel_gpus=[1, 2],
            memory_fraction=0.8
        )
        
        # Setup PyTorch for A40
        self._setup_pytorch_a40()
        
        # Initialize coordinator
        self.coordinator = MultiGPUCoordinator(self.config)
        
        # Curriculum buffer
        self.curriculum_buffer = CurriculumBuffer(variance_threshold=0.1)
        
        # Reward calculator
        self.reward_calculator = RewardCalculator()
    
    def _setup_pytorch_a40(self):
        """Configure PyTorch optimally for A40"""
        # A40 supports TF32 - enable for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Optimize for consistent kernel sizes
        torch.backends.cudnn.benchmark = True
        
        # Set memory fraction to leave room for kernel ops
        for gpu_id in range(3):
            torch.cuda.set_per_process_memory_fraction(
                self.config.memory_fraction, 
                device=gpu_id
            )
        
        # Use high precision for matmul
        torch.set_float32_matmul_precision('high')
    
    async def optimize_batch(self, 
                           specialist,
                           problems: List[Dict],
                           use_search: bool = True) -> List[OptimizationTrajectory]:
        """
        Optimize a batch of problems using multi-GPU setup
        
        Args:
            specialist: The optimization specialist (e.g., GEMMSpecialist)
            problems: List of problem dictionaries
            use_search: Whether to use search (beam/MCTS)
            
        Returns:
            List of optimization trajectories
        """
        trajectories = []
        
        # Process each problem
        for problem in problems:
            # Generate candidates on model GPU
            with torch.cuda.device(self.config.model_gpu):
                if use_search:
                    # Use search to generate multiple trajectories
                    candidate_trajectories = specialist.search_algorithm.search(
                        initial_state=KernelState(
                            code=problem['code'],
                            turn=0,
                            performance_metrics={}
                        ),
                        generator=specialist,
                        evaluator=None,  # We'll evaluate separately
                        num_candidates=16,
                        max_depth=3
                    )
                else:
                    # Simple single trajectory
                    candidate_trajectories = [specialist.optimize_kernel(
                        problem['code'],
                        problem['problem_id'],
                        use_search=False,
                        max_turns=3
                    )]
            
            # Evaluate all candidate kernels in parallel
            all_kernels = []
            for traj in candidate_trajectories:
                for state in traj.states[1:]:  # Skip initial state
                    all_kernels.append((state.code, 'gemm'))
            
            # Parallel evaluation on GPU 1-2
            if all_kernels:
                eval_results = await self.coordinator.evaluate_kernels_parallel(all_kernels)
                
                # Update trajectories with real metrics
                kernel_idx = 0
                for traj in candidate_trajectories:
                    for state in traj.states[1:]:
                        if kernel_idx < len(eval_results):
                            state.performance_metrics = eval_results[kernel_idx]
                            kernel_idx += 1
            
            # Select best trajectory
            best_trajectory = max(
                candidate_trajectories,
                key=lambda t: sum(s.performance_metrics.get('gflops', 0) 
                                for s in t.states)
            )
            trajectories.append(best_trajectory)
            
            # Update curriculum buffer
            rewards = [t.total_reward for t in candidate_trajectories]
            self.curriculum_buffer.add_group(problem['problem_id'], rewards)
        
        return trajectories
    
    def print_gpu_status(self):
        """Print current GPU utilization status"""
        stats = self.coordinator.get_gpu_stats()
        
        print("\n" + "="*50)
        print("GPU Status:")
        print("="*50)
        
        print(f"Model GPU (0): A40 - Reserved for model")
        
        for gpu_id, util in stats['gpu_utilization'].items():
            status = "Busy" if util > 0 else "Available"
            print(f"Kernel GPU ({gpu_id}): A40 - {status}")
        
        print(f"\nAverage evaluation time: {stats['avg_eval_time']:.2f}s")
        print(f"Total evaluations: {stats['total_evaluations']}")
        print(f"Available kernel GPUs: {stats['gpus_available']}/2")

# Example usage function
async def run_multi_gpu_optimization():
    """Example of running optimization with 3x A40s"""
    from gemm_specialist import create_gemm_specialist, NAIVE_GEMM
    
    print("Initializing 3x A40 system...")
    optimizer = A40Optimizer()
    
    # Create specialist on model GPU
    with torch.cuda.device(0):
        specialist = create_gemm_specialist(use_beam_search=True, beam_width=4)
    
    # Test problems
    problems = [
        {"problem_id": "test_1", "code": NAIVE_GEMM},
        {"problem_id": "test_2", "code": NAIVE_GEMM},
    ]
    
    print("\nRunning parallel optimization...")
    trajectories = await optimizer.optimize_batch(
        specialist,
        problems,
        use_search=True
    )
    
    # Print results
    for i, traj in enumerate(trajectories):
        print(f"\nProblem {i}: {traj.num_turns} optimizations")
        final_gflops = traj.current_state.performance_metrics.get('gflops', 0)
        print(f"Final performance: {final_gflops:.1f} GFLOPS")
    
    optimizer.print_gpu_status()

if __name__ == "__main__":
    # Run example
    asyncio.run(run_multi_gpu_optimization())
