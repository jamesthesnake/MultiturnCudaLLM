"""
KernelBench Integration for CUDA RL System
Connects the RL system with ScalingIntelligence/KernelBench
"""

import subprocess
import json
import tempfile
import os
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
from pathlib import Path

class KernelBenchIntegration:
    """
    Integration with ScalingIntelligence/KernelBench
    Assumes KernelBench is installed and available
    """
    
    def __init__(self, kernelbench_path: str = None):
        """
        Initialize KernelBench integration
        
        Args:
            kernelbench_path: Path to KernelBench installation
        """
        self.kernelbench_path = kernelbench_path or self._find_kernelbench()
        
        if not os.path.exists(self.kernelbench_path):
            raise ValueError(f"KernelBench not found at {self.kernelbench_path}")
        
        # Load available benchmarks
        self.available_benchmarks = self._load_available_benchmarks()
    
    def _find_kernelbench(self) -> str:
        """Try to find KernelBench installation"""
        # Common locations
        possible_paths = [
            os.path.expanduser("~/KernelBench"),
            "/opt/KernelBench",
            "./KernelBench",
            "../KernelBench"
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "benchmark.py")):
                return path
        
        raise ValueError("Could not find KernelBench installation")
    
    def _load_available_benchmarks(self) -> Dict[str, List[str]]:
        """Load list of available benchmarks from KernelBench"""
        benchmarks = {
            'gemm': ['gemm_fp32', 'gemm_fp16', 'gemm_int8'],
            'conv': ['conv2d_fp32', 'conv2d_fp16'],
            'attention': ['attention_fp32', 'attention_fp16'],
            'reduction': ['reduce_sum', 'reduce_max'],
            'scan': ['prefix_sum', 'cumsum'],
            'fft': ['fft_1d', 'fft_2d']
        }
        return benchmarks
    
    def evaluate_kernel(self, 
                       kernel_code: str,
                       benchmark_name: str,
                       problem_size: Optional[Dict] = None) -> Dict[str, float]:
        """
        Evaluate a kernel using KernelBench
        
        Args:
            kernel_code: CUDA kernel code
            benchmark_name: Name of the benchmark (e.g., 'gemm_fp32')
            problem_size: Optional problem size parameters
            
        Returns:
            Dictionary with performance metrics
        """
        
        # Write kernel to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(kernel_code)
            kernel_path = f.name
        
        try:
            # Prepare command
            cmd = [
                'python',
                os.path.join(self.kernelbench_path, 'benchmark.py'),
                '--kernel', kernel_path,
                '--benchmark', benchmark_name,
                '--output-format', 'json'
            ]
            
            # Add problem size if specified
            if problem_size:
                for key, value in problem_size.items():
                    cmd.extend([f'--{key}', str(value)])
            
            # Run benchmark
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if result.returncode != 0:
                return {
                    'compiles': False,
                    'correct': False,
                    'error_msg': result.stderr,
                    'gflops': 0.0,
                    'bandwidth_gb_s': 0.0,
                    'kernel_time_ms': float('inf')
                }
            
            # Parse JSON output
            output = json.loads(result.stdout)
            
            # Extract metrics
            metrics = {
                'compiles': True,
                'correct': output.get('correct', True),
                'gflops': output.get('gflops', 0.0),
                'bandwidth_gb_s': output.get('bandwidth_gb_s', 0.0),
                'kernel_time_ms': output.get('kernel_time_ms', 0.0),
                'occupancy': output.get('occupancy', 0.0),
                'registers_per_thread': output.get('registers_per_thread', 0),
                'shared_memory_bytes': output.get('shared_memory_bytes', 0),
                'error_msg': output.get('error_msg', '')
            }
            
            # Add derived metrics
            if benchmark_name.startswith('gemm'):
                # For GEMM, calculate efficiency vs theoretical peak
                # Assuming V100 with ~15 TFLOPS peak
                theoretical_peak = 15000  # GFLOPS
                metrics['efficiency'] = metrics['gflops'] / theoretical_peak
            
            return metrics
            
        except subprocess.TimeoutExpired:
            return {
                'compiles': True,
                'correct': False,
                'error_msg': 'Benchmark timeout',
                'gflops': 0.0,
                'bandwidth_gb_s': 0.0,
                'kernel_time_ms': float('inf')
            }
        except Exception as e:
            return {
                'compiles': False,
                'correct': False,
                'error_msg': str(e),
                'gflops': 0.0,
                'bandwidth_gb_s': 0.0,
                'kernel_time_ms': float('inf')
            }
        finally:
            # Cleanup
            if os.path.exists(kernel_path):
                os.remove(kernel_path)
    
    def get_baseline_kernels(self, benchmark_type: str) -> List[Tuple[str, str]]:
        """
        Get baseline kernel implementations from KernelBench
        
        Returns:
            List of (kernel_code, kernel_name) tuples
        """
        baseline_dir = os.path.join(self.kernelbench_path, 'baselines', benchmark_type)
        
        if not os.path.exists(baseline_dir):
            return []
        
        kernels = []
        
        for kernel_file in os.listdir(baseline_dir):
            if kernel_file.endswith('.cu'):
                kernel_path = os.path.join(baseline_dir, kernel_file)
                with open(kernel_path, 'r') as f:
                    kernel_code = f.read()
                
                kernel_name = os.path.splitext(kernel_file)[0]
                kernels.append((kernel_code, kernel_name))
        
        return kernels
    
    def profile_kernel(self, 
                      kernel_code: str,
                      benchmark_name: str,
                      profile_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Profile a kernel using nvprof or ncu
        
        Args:
            kernel_code: CUDA kernel code
            benchmark_name: Benchmark name
            profile_metrics: List of metrics to collect
            
        Returns:
            Dictionary with profiling results
        """
        if profile_metrics is None:
            profile_metrics = [
                'sm_efficiency',
                'achieved_occupancy',
                'gld_throughput',
                'gst_throughput',
                'shared_load_throughput',
                'shared_store_throughput',
                'l2_cache_throughput',
                'dram_throughput'
            ]
        
        # Write kernel to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(kernel_code)
            kernel_path = f.name
        
        try:
            # Use ncu (NVIDIA Nsight Compute) for profiling
            metrics_str = ','.join(profile_metrics)
            cmd = [
                'ncu',
                '--metrics', metrics_str,
                '--csv',
                'python',
                os.path.join(self.kernelbench_path, 'benchmark.py'),
                '--kernel', kernel_path,
                '--benchmark', benchmark_name,
                '--iterations', '10'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                return {'error': result.stderr}
            
            # Parse CSV output
            lines = result.stdout.strip().split('\n')
            header = lines[0].split(',')
            values = lines[1].split(',') if len(lines) > 1 else []
            
            profile_data = {}
            for i, metric in enumerate(header):
                if i < len(values):
                    try:
                        profile_data[metric] = float(values[i])
                    except ValueError:
                        profile_data[metric] = values[i]
            
            return profile_data
            
        except Exception as e:
            return {'error': str(e)}
        finally:
            if os.path.exists(kernel_path):
                os.remove(kernel_path)
    
    def generate_problem_sizes(self, benchmark_type: str) -> List[Dict[str, int]]:
        """
        Generate a range of problem sizes for testing
        
        Args:
            benchmark_type: Type of benchmark (e.g., 'gemm')
            
        Returns:
            List of problem size configurations
        """
        if benchmark_type == 'gemm':
            # Matrix sizes for GEMM
            sizes = []
            for size in [128, 256, 512, 1024, 2048, 4096]:
                sizes.append({
                    'M': size,
                    'N': size,
                    'K': size
                })
            # Also add some rectangular matrices
            sizes.extend([
                {'M': 1024, 'N': 768, 'K': 512},
                {'M': 2048, 'N': 512, 'K': 1024},
                {'M': 4096, 'N': 1024, 'K': 2048}
            ])
            return sizes
            
        elif benchmark_type == 'conv':
            # Convolution sizes
            sizes = []
            for batch in [1, 8, 32]:
                for size in [28, 56, 112, 224]:
                    sizes.append({
                        'batch': batch,
                        'height': size,
                        'width': size,
                        'channels': 64,
                        'filters': 64,
                        'kernel_size': 3
                    })
            return sizes
            
        elif benchmark_type == 'attention':
            # Attention sizes
            sizes = []
            for seq_len in [128, 256, 512, 1024]:
                for hidden in [256, 512, 768]:
                    sizes.append({
                        'batch': 16,
                        'seq_len': seq_len,
                        'hidden_dim': hidden,
                        'num_heads': 8
                    })
            return sizes
            
        else:
            # Default sizes
            return [
                {'size': 1024},
                {'size': 4096},
                {'size': 16384},
                {'size': 65536}
            ]

class KernelBenchDataset:
    """
    Dataset of kernels from KernelBench for training
    """
    
    def __init__(self, kernelbench: KernelBenchIntegration, benchmark_types: List[str] = None):
        self.kernelbench = kernelbench
        self.benchmark_types = benchmark_types or ['gemm', 'conv', 'reduction']
        self.kernels = self._load_kernels()
    
    def _load_kernels(self) -> List[Dict]:
        """Load all baseline kernels from KernelBench"""
        kernels = []
        
        for benchmark_type in self.benchmark_types:
            baseline_kernels = self.kernelbench.get_baseline_kernels(benchmark_type)
            
            for kernel_code, kernel_name in baseline_kernels:
                # Create multiple problem instances
                problem_sizes = self.kernelbench.generate_problem_sizes(benchmark_type)
                
                for i, size_config in enumerate(problem_sizes[:3]):  # Limit to 3 sizes per kernel
                    kernels.append({
                        'problem_id': f'{benchmark_type}_{kernel_name}_size{i}',
                        'code': kernel_code,
                        'type': benchmark_type,
                        'benchmark_name': f'{benchmark_type}_fp32',
                        'problem_size': size_config,
                        'source': 'kernelbench'
                    })
        
        return kernels
    
    def __len__(self):
        return len(self.kernels)
    
    def __getitem__(self, idx):
        return self.kernels[idx]

# ===== Usage Example =====

def test_kernelbench_integration():
    """Test the KernelBench integration"""
    
    try:
        # Initialize integration
        kb = KernelBenchIntegration()
        print(f"KernelBench found at: {kb.kernelbench_path}")
        print(f"Available benchmarks: {kb.available_benchmarks}")
        
        # Test with a simple GEMM kernel
        test_kernel = """
__global__ void simple_gemm(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""
        
        # Evaluate kernel
        print("\nEvaluating kernel...")
        metrics = kb.evaluate_kernel(
            test_kernel, 
            'gemm_fp32',
            problem_size={'M': 1024, 'N': 1024, 'K': 1024}
        )
        
        print("Evaluation results:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Get baseline kernels
        print("\nLoading baseline kernels...")
        baselines = kb.get_baseline_kernels('gemm')
        print(f"Found {len(baselines)} baseline GEMM kernels")
        
        # Create dataset
        print("\nCreating dataset...")
        dataset = KernelBenchDataset(kb, benchmark_types=['gemm'])
        print(f"Dataset contains {len(dataset)} kernel instances")
        
        # Show sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample kernel:")
            print(f"  Problem ID: {sample['problem_id']}")
            print(f"  Type: {sample['type']}")
            print(f"  Problem size: {sample['problem_size']}")
            
    except Exception as e:
        print(f"KernelBench integration test failed: {e}")
        print("Make sure KernelBench is installed from https://github.com/ScalingIntelligence/KernelBench")

if __name__ == "__main__":
    test_kernelbench_integration()
