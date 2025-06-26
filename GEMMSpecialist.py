"""
GEMM Optimization Specialist
Implements progressive optimization: Naive → Tiling → Shared Memory → Bank Conflicts → Tensor Cores
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import List, Tuple, Dict, Optional
import numpy as np

from cuda_rl_system import (
    CUDAOptimizationSpecialist, KernelState, OptimizationType,
    KernelEvaluator, SearchAlgorithm, RewardCalculator
)

class GEMMSpecialist(CUDAOptimizationSpecialist):
    """
    Specialist for optimizing GEMM (General Matrix Multiplication) kernels
    Progressive optimization strategy across turns
    """
    
    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-coder-1.3b-base",
                 evaluator: Optional[KernelEvaluator] = None,
                 search_algorithm: Optional[SearchAlgorithm] = None,
                 device: str = "cuda"):
        
        # GEMM-specific optimization types in progressive order
        optimization_types = [
            OptimizationType.MEMORY_COALESCING,
            OptimizationType.SHARED_MEMORY,
            OptimizationType.BANK_CONFLICT,
            OptimizationType.THREAD_CONFIG,
            OptimizationType.REGISTER_BLOCKING,
            OptimizationType.TENSOR_CORES
        ]
        
        # GEMM-specific reward bonuses
        reward_calculator = RewardCalculator(
            turn_bonuses={
                1: {
                    'memory_coalescing': 0.15,
                    'tiling': 0.2,
                    'thread_config': 0.1
                },
                2: {
                    'shared_memory': 0.3,
                    'synchronization': 0.1,
                    'double_buffering': 0.15
                },
                3: {
                    'bank_conflict': 0.25,
                    'register_blocking': 0.2,
                    'vectorized_loads': 0.15
                },
                4: {
                    'tensor_cores': 0.4,
                    'wmma_optimization': 0.3
                }
            }
        )
        
        super().__init__(
            model_name=model_name,
            optimization_types=optimization_types,
            evaluator=evaluator or KernelEvaluator(),
            search_algorithm=search_algorithm,
            reward_calculator=reward_calculator
        )
        
        # Load model and tokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        
        # GEMM-specific prompts for each turn
        self.turn_prompts = {
            1: self._get_turn1_prompt,
            2: self._get_turn2_prompt,
            3: self._get_turn3_prompt,
            4: self._get_turn4_prompt
        }
    
    def generate_optimizations(self,
                             state: KernelState,
                             num_candidates: int = 8,
                             turn: int = 1) -> List[Tuple[str, str, str]]:
        """
        Generate optimization candidates for the current turn
        """
        # Get turn-specific prompt
        prompt_fn = self.turn_prompts.get(turn, self._get_turn1_prompt)
        base_prompt = prompt_fn(state.code)
        
        candidates = []
        
        # Generate multiple candidates with different sampling
        temperatures = np.linspace(0.7, 1.2, num_candidates)
        
        for i, temp in enumerate(temperatures):
            try:
                # Add some variation to the prompt
                variation = f"\n// Approach {i+1}: Focus on {self._get_focus_area(turn, i)}"
                prompt = base_prompt + variation
                
                # Generate optimization
                code, reasoning, opt_type = self._generate_single_optimization(
                    prompt, 
                    temperature=temp,
                    turn=turn
                )
                
                if code and self._is_valid_cuda_code(code):
                    candidates.append((code, reasoning, opt_type))
                    
            except Exception as e:
                print(f"Error generating candidate {i}: {e}")
                continue
        
        return candidates
    
    def _generate_single_optimization(self, 
                                    prompt: str, 
                                    temperature: float,
                                    turn: int) -> Tuple[str, str, str]:
        """Generate a single optimization using the model"""
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract code, reasoning, and optimization type
        code = self._extract_code(generated_text)
        reasoning = self._extract_reasoning(generated_text)
        opt_type = self._determine_optimization_type(code, turn)
        
        return code, reasoning, opt_type
    
    def _get_turn1_prompt(self, code: str) -> str:
        """Turn 1: Basic optimizations - tiling, coalescing, thread configuration"""
        return f"""<|thinking_start|>
The user has provided a GEMM kernel that needs optimization. This is turn 1, so I should focus on:
1. Memory coalescing - ensure consecutive threads access consecutive memory
2. Basic tiling to improve cache usage
3. Optimal thread block configuration

Let me analyze the current kernel and apply these optimizations.
<|thinking_end|>

Optimize this GEMM kernel for better performance. Focus on memory coalescing and basic tiling.

Current kernel:
```cuda
{code}
```

Optimized kernel with explanation:
```cuda
// Optimization reasoning:
// 1. Added tiling with TILE_SIZE for better cache usage
// 2. Ensured coalesced memory access patterns
// 3. Optimized thread block dimensions
"""

    def _get_turn2_prompt(self, code: str) -> str:
        """Turn 2: Shared memory optimization"""
        return f"""<|thinking_start|>
This is turn 2 of optimization. The kernel already has basic tiling. Now I need to:
1. Add shared memory usage to reduce global memory access
2. Implement proper synchronization with __syncthreads()
3. Consider double buffering if beneficial
4. Optimize the loading pattern into shared memory

This is a critical optimization that usually provides significant speedup.
<|thinking_end|>

Further optimize this GEMM kernel by adding shared memory usage.

Current kernel:
```cuda
{code}
```

Optimized kernel with shared memory:
```cuda
// Optimization reasoning:
// 1. Added shared memory tiles for both matrices A and B
// 2. Collaborative loading with coalesced access
// 3. Proper synchronization to avoid race conditions
// 4. Reduced global memory bandwidth by ~2x
"""

    def _get_turn3_prompt(self, code: str) -> str:
        """Turn 3: Bank conflict resolution and register blocking"""
        return f"""<|thinking_start|>
This is turn 3. The kernel now uses shared memory. I need to:
1. Analyze and fix any bank conflicts in shared memory access
2. Add padding if necessary to avoid conflicts
3. Implement register blocking for better register usage
4. Consider vectorized loads (float2 or float4)

Bank conflicts can severely impact shared memory performance.
<|thinking_end|>

Optimize this GEMM kernel by resolving bank conflicts and adding register-level optimizations.

Current kernel:
```cuda
{code}
```

Optimized kernel with bank conflict resolution:
```cuda
// Optimization reasoning:
// 1. Added padding to shared memory to avoid bank conflicts
// 2. Implemented 2x2 register blocking for better ILP
// 3. Used vectorized loads where possible
// 4. Reordered operations to hide latency
"""

    def _get_turn4_prompt(self, code: str) -> str:
        """Turn 4: Tensor Core optimization (if applicable)"""
        return f"""<|thinking_start|>
This is turn 4, the final optimization. For modern GPUs (Volta+), I should:
1. Check if we can use Tensor Cores via WMMA API
2. Ensure data layout is compatible (16x16x16 tiles)
3. Use half precision where appropriate
4. Implement the wmma::mma_sync operation

Tensor Cores can provide up to 8x speedup for GEMM operations.
<|thinking_end|>

Apply Tensor Core optimizations to this GEMM kernel if the GPU supports it.

Current kernel:
```cuda
{code}
```

Tensor Core optimized kernel:
```cuda
// Optimization reasoning:
// 1. Converted to use WMMA API for Tensor Cores
// 2. Adjusted tile sizes to 16x16x16 for Tensor Core compatibility
// 3. Added half precision support where appropriate
// 4. Achieved theoretical peak performance improvement
"""

    def _get_focus_area(self, turn: int, variation: int) -> str:
        """Get specific focus area for variation in generation"""
        focus_areas = {
            1: ["row-major access", "column-major access", "diagonal tiling", "Z-order pattern"],
            2: ["1D shared memory", "2D shared memory", "double buffering", "prefetching"],
            3: ["padding strategy", "permuted access", "conflict-free patterns", "skewed indexing"],
            4: ["HMMA instructions", "mixed precision", "fragment accumulators", "cooperative groups"]
        }
        
        areas = focus_areas.get(turn, ["general optimization"])
        return areas[variation % len(areas)]
    
    def _extract_code(self, generated_text: str) -> str:
        """Extract CUDA code from generated text"""
        # Look for code blocks
        code_pattern = r'```cuda\n(.*?)```'
        matches = re.findall(code_pattern, generated_text, re.DOTALL)
        
        if matches:
            # Take the last code block (should be the optimized version)
            return matches[-1].strip()
        
        # Fallback: look for __global__ function
        kernel_pattern = r'(__global__.*?(?=__global__|$))'
        matches = re.findall(kernel_pattern, generated_text, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
        
        return ""
    
    def _extract_reasoning(self, generated_text: str) -> str:
        """Extract optimization reasoning from generated text"""
        # Look for reasoning section
        reasoning_pattern = r'//\s*(?:Optimization reasoning|Reasoning):?\s*((?:.*\n)*?)(?=```|__global__|$)'
        matches = re.findall(reasoning_pattern, generated_text, re.MULTILINE)
        
        if matches:
            return matches[0].strip()
        
        # Fallback: extract comments
        comment_pattern = r'//\s*\d+\.\s*(.*?)(?=\n|$)'
        comments = re.findall(comment_pattern, generated_text)
        
        if comments:
            return " ".join(comments)
        
        return "Applied optimizations to improve performance"
    
    def _determine_optimization_type(self, code: str, turn: int) -> str:
        """Determine what type of optimization was applied"""
        code_lower = code.lower()
        
        # Check for specific patterns
        if "wmma" in code_lower or "tensor" in code_lower:
            return OptimizationType.TENSOR_CORES.value
        elif "__shared__" in code_lower:
            if "padding" in code_lower or "conflict" in code_lower:
                return OptimizationType.BANK_CONFLICT.value
            else:
                return OptimizationType.SHARED_MEMORY.value
        elif "tile" in code_lower or "block" in code_lower:
            return OptimizationType.MEMORY_COALESCING.value
        elif "register" in code_lower or "ilp" in code_lower:
            return OptimizationType.REGISTER_BLOCKING.value
        else:
            # Default based on turn
            defaults = {
                1: OptimizationType.MEMORY_COALESCING.value,
                2: OptimizationType.SHARED_MEMORY.value,
                3: OptimizationType.BANK_CONFLICT.value,
                4: OptimizationType.TENSOR_CORES.value
            }
            return defaults.get(turn, OptimizationType.THREAD_CONFIG.value)
    
    def _is_valid_cuda_code(self, code: str) -> bool:
        """Basic validation of CUDA code"""
        if not code:
            return False
        
        # Check for essential CUDA elements
        required_patterns = [
            r'__global__',  # Kernel declaration
            r'void\s+\w+\s*\(',  # Function signature
            r'threadIdx|blockIdx|blockDim|gridDim'  # CUDA built-ins
        ]
        
        for pattern in required_patterns:
            if not re.search(pattern, code):
                return False
        
        # Check for balanced braces
        if code.count('{') != code.count('}'):
            return False
        
        return True

# ===== Example Usage Functions =====

def create_gemm_specialist(use_beam_search: bool = True, 
                          beam_width: int = 8,
                          kernelbench_path: Optional[str] = None) -> GEMMSpecialist:
    """Create a GEMM specialist with specified search algorithm"""
    
    evaluator = KernelEvaluator(kernelbench_path=kernelbench_path)
    
    if use_beam_search:
        from cuda_rl_system import BeamSearch
        search_algorithm = BeamSearch(beam_width=beam_width)
    else:
        from cuda_rl_system import MCTS
        search_algorithm = MCTS(num_simulations=100)
    
    specialist = GEMMSpecialist(
        evaluator=evaluator,
        search_algorithm=search_algorithm
    )
    
    return specialist

def optimize_gemm_kernel(kernel_code: str, 
                        specialist: GEMMSpecialist,
                        problem_id: str = "gemm_001",
                        use_search: bool = True,
                        max_turns: int = 4) -> Dict:
    """
    Optimize a GEMM kernel and return results
    """
    print(f"Optimizing GEMM kernel (search={'enabled' if use_search else 'disabled'})...")
    
    trajectory = specialist.optimize_kernel(
        initial_code=kernel_code,
        problem_id=problem_id,
        use_search=use_search,
        max_turns=max_turns
    )
    
    # Prepare results
    results = {
        'trajectory': trajectory,
        'initial_performance': trajectory.states[0].performance_metrics,
        'final_performance': trajectory.current_state.performance_metrics,
        'total_reward': trajectory.total_reward,
        'improvements': []
    }
    
    # Calculate improvements at each turn
    for i in range(1, len(trajectory.states)):
        prev_state = trajectory.states[i-1]
        curr_state = trajectory.states[i]
        
        improvement = {
            'turn': i,
            'optimization': curr_state.optimization_applied,
            'gflops_before': prev_state.performance_metrics.get('gflops', 0),
            'gflops_after': curr_state.performance_metrics.get('gflops', 0),
            'speedup': curr_state.performance_metrics.get('gflops', 0) / max(prev_state.performance_metrics.get('gflops', 1), 1),
            'reasoning': curr_state.reasoning
        }
        results['improvements'].append(improvement)
    
    return results

# ===== Sample GEMM Kernels for Testing =====

NAIVE_GEMM = """
__global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
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

TILED_GEMM = """
#define TILE_SIZE 16

__global__ void gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (row < M && tile * TILE_SIZE + tx < K) {
            // Load A element
        }
        if (col < N && tile * TILE_SIZE + ty < K) {
            // Load B element
        }
        __syncthreads();
        
        // Compute partial dot product
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"""

if __name__ == "__main__":
    # Example usage
    specialist = create_gemm_specialist(use_beam_search=True, beam_width=8)
    
    # Test with naive GEMM
    results = optimize_gemm_kernel(
        NAIVE_GEMM, 
        specialist, 
        problem_id="test_gemm_001",
        use_search=True,
        max_turns=3
    )
    
    print("\nOptimization Results:")
    print(f"Initial GFLOPS: {results['initial_performance'].get('gflops', 0):.2f}")
    print(f"Final GFLOPS: {results['final_performance'].get('gflops', 0):.2f}")
    print(f"Total speedup: {results['final_performance'].get('gflops', 0) / max(results['initial_performance'].get('gflops', 1), 1):.2f}x")
    
    print("\nOptimization trajectory:")
    for imp in results['improvements']:
        print(f"Turn {imp['turn']}: {imp['optimization']} - {imp['speedup']:.2f}x speedup")
        print(f"  Reasoning: {imp['reasoning'][:100]}...")
