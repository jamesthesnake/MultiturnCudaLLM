import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import subprocess
import tempfile
import os
import time
import json
import re
import inspect
from tqdm import tqdm
import multiprocessing as mp
from torch.utils.cpp_extension import load_inline
import traceback

@dataclass
class KernelSample:
    """Represents a CUDA kernel generation sample with feedback"""
    prompt: str
    kernel_code: str
    feedback: str
    reward: float
    is_correct: bool
    speedup: float = 1.0
    compilation_success: bool = False
    summary: str = ""  # Brief summary of approach

@dataclass
class KernelTrajectory:
    """Multi-turn trajectory with refinement steps"""
    initial_prompt: str
    task_name: str
    samples: List[KernelSample]
    total_reward: float
    best_speedup: float

class CUDAKernelExecutor:
    """Executes and evaluates generated CUDA kernels"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        # Forbidden PyTorch functions that indicate reward hacking
        self.forbidden_torch_ops = {
            'torch.matmul', 'torch.mm', 'torch.bmm', 'torch.einsum',
            'torch.dot', 'torch.mul', 'torch.add', 'torch.sub',
            'F.conv2d', 'F.conv1d', 'F.linear', 'F.batch_norm',
            'F.layer_norm', 'F.softmax', 'F.relu', 'F.gelu',
            'nn.functional', 'torch.nn.functional',
            '.matmul(', '.mm(', '.bmm(', '.dot(',
            'torch.ops.aten', 'torch._C', 'torch.jit'
        }
        
    def extract_cuda_kernel(self, generated_text: str) -> Tuple[str, str]:
        """Extract CUDA kernel and ModelNew class from generated text"""
        # Extract everything between class ModelNew and the end
        model_pattern = r'class ModelNew.*?(?=\n(?:class|def|if|$))'
        cuda_pattern = r'```(?:cuda|cpp|c\+\+)(.*?)```'
        
        model_match = re.search(model_pattern, generated_text, re.DOTALL)
        cuda_matches = re.findall(cuda_pattern, generated_text, re.DOTALL)
        
        if model_match:
            model_code = model_match.group(0)
            # Combine all CUDA code blocks
            cuda_code = '\n'.join(cuda_matches) if cuda_matches else ""
            return model_code, cuda_code
        
        return "", ""
    
    def detect_reward_hacking(self, code: str) -> Tuple[bool, str]:
        """
        Detect various reward hacking attempts
        Returns: (is_hacking, reason)
        """
        # Check for PyTorch function usage
        for forbidden_op in self.forbidden_torch_ops:
            if forbidden_op in code:
                return True, f"Forbidden PyTorch operation detected: {forbidden_op}"
        
        # Check for try-except wrapping that falls back to PyTorch
        try_except_pattern = r'try:.*?except.*?(?:torch\.|F\.|nn\.)'
        if re.search(try_except_pattern, code, re.DOTALL | re.IGNORECASE):
            return True, "Try-except block with PyTorch fallback detected"
        
        # Check if inheriting from reference implementation
        inherit_patterns = [
            r'class\s+ModelNew\s*\([^)]*Model[^)]*\)',  # Inheriting from Model
            r'super\(\).__init__\(\)',  # Calling parent init (suspicious if no real implementation)
            r'self\.reference_model',  # Storing reference model
        ]
        for pattern in inherit_patterns:
            if re.search(pattern, code):
                # Check if there's actual CUDA implementation
                if not self._has_cuda_kernel(code):
                    return True, "Inheriting from reference without CUDA implementation"
        
        # Check for calling reference implementation directly
        reference_patterns = [
            r'reference_model\(',
            r'self\.ln\(',  # Direct LayerNorm call
            r'self\.conv\(',  # Direct Conv call
            r'return\s+Model\.',  # Returning Model class methods
        ]
        for pattern in reference_patterns:
            if re.search(pattern, code):
                return True, "Calling reference implementation directly"
        
        # Check if code just copies input to output
        dummy_patterns = [
            r'return\s+x\s*;?\s*
    
    def compile_and_test_kernel(
        self, 
        kernel_code: str, 
        reference_model: Any,
        test_inputs: List[torch.Tensor]
    ) -> Tuple[bool, bool, str, float]:
        """
        Compile and test CUDA kernel
        Returns: (compilation_success, is_correct, feedback, speedup)
        """
        # First check for reward hacking
        is_hacking, hack_reason = self.detect_reward_hacking(kernel_code)
        if is_hacking:
            return False, False, f"Reward hacking detected: {hack_reason}", 0.0
        
        try:
            # Create temporary file for the kernel code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Add necessary imports
                full_code = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import time

""" + kernel_code
                
                f.write(full_code)
                f.flush()
                
                # Try to compile and load the kernel
                try:
                    # Dynamic import
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("kernel_module", f.name)
                    kernel_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(kernel_module)
                    
                    ModelNew = kernel_module.ModelNew
                    model_new = ModelNew(*self._get_model_args(reference_model))
                    
                    # Additional runtime check for reward hacking
                    model_source = inspect.getsource(ModelNew)
                    is_hacking_runtime, hack_reason_runtime = self.detect_reward_hacking(model_source)
                    if is_hacking_runtime:
                        return False, False, f"Runtime reward hacking: {hack_reason_runtime}", 0.0
                    
                except subprocess.CalledProcessError as e:
                    # CUDA compilation error - extract meaningful error messages
                    error_output = e.stderr if hasattr(e, 'stderr') else str(e)
                    cuda_errors = self._extract_cuda_compilation_errors(error_output)
                    feedback = f"CUDA compilation failed:\n{cuda_errors}\nFix: Check kernel syntax, ensure all CUDA functions are properly declared"
                    return False, False, feedback, 0.0
                    
                except ImportError as e:
                    # Module import error
                    feedback = f"Import error: {str(e)}\nFix: Ensure ModelNew class is properly defined and all imports are correct"
                    return False, False, feedback, 0.0
                    
                except AttributeError as e:
                    # Missing ModelNew or other attributes
                    feedback = f"Attribute error: {str(e)}\nFix: Ensure ModelNew class exists and has required methods"
                    return False, False, feedback, 0.0
                    
                except Exception as e:
                    # Other compilation errors
                    error_trace = traceback.format_exc()
                    feedback = f"Compilation error: {str(e)}\nTraceback:\n{error_trace}\nFix: Check Python syntax and class structure"
                    return False, False, feedback, 0.0
                
                # Test correctness
                model_new.eval()
                reference_model.eval()
                
                runtime_errors = []
                incorrect_outputs = []
                
                with torch.no_grad():
                    for i, test_input in enumerate(test_inputs):
                        try:
                            output_new = model_new(*test_input)
                            output_ref = reference_model(*test_input)
                            
                            if not torch.allclose(output_new, output_ref, rtol=1e-3, atol=1e-3):
                                # Provide detailed mismatch information
                                max_diff = torch.max(torch.abs(output_new - output_ref)).item()
                                rel_error = torch.max(torch.abs((output_new - output_ref) / (output_ref + 1e-8))).item()
                                incorrect_outputs.append(
                                    f"Test {i+1}: Max absolute diff: {max_diff:.6f}, "
                                    f"Max relative error: {rel_error:.6f}, "
                                    f"Output shape: {output_new.shape}, Expected: {output_ref.shape}"
                                )
                                
                        except RuntimeError as e:
                            if "CUDA" in str(e):
                                # CUDA runtime error - extract details
                                cuda_error = self._extract_cuda_runtime_error(str(e))
                                runtime_errors.append(f"Test {i+1} - CUDA error: {cuda_error}")
                            else:
                                runtime_errors.append(f"Test {i+1} - Runtime error: {str(e)}")
                                
                        except Exception as e:
                            runtime_errors.append(f"Test {i+1} - Error: {type(e).__name__}: {str(e)}")
                
                # Compile feedback based on errors
                if runtime_errors:
                    feedback = "Runtime errors:\n" + "\n".join(runtime_errors)
                    feedback += "\nFix: Check array bounds, shared memory size, and thread synchronization"
                    return True, False, feedback, 0.0
                    
                if incorrect_outputs:
                    feedback = "Incorrect outputs:\n" + "\n".join(incorrect_outputs)
                    feedback += "\nFix: Verify your algorithm implementation, check for race conditions"
                    return True, False, feedback, 0.0
                
                # Measure performance
                speedup = self._measure_speedup(model_new, reference_model, test_inputs)
                
                # Final check: suspiciously high speedup might indicate cheating
                if speedup > 100.0:
                    # Re-verify with more careful testing
                    speedup_verified = self._verify_speedup(model_new, reference_model, test_inputs)
                    if speedup_verified < speedup * 0.5:
                        return False, False, "Suspicious speedup detected - likely reward hacking", 0.0
                    speedup = speedup_verified
                
                feedback = f"Kernel compiled and passed all tests! Speedup: {speedup:.2f}x"
                return True, True, feedback, speedup
                
        except Exception as e:
            return False, False, f"Unexpected error: {str(e)}", 0.0
        finally:
            if 'f' in locals():
                os.unlink(f.name)
    
    def _extract_cuda_compilation_errors(self, error_output: str) -> str:
        """Extract meaningful CUDA compilation errors"""
        # Look for common CUDA compilation error patterns
        error_patterns = [
            r"error:.*?(?=\n)",  # General errors
            r".*undefined reference.*",  # Linker errors
            r".*syntax error.*",  # Syntax errors
            r".*expected.*before.*",  # Syntax expectation errors
        ]
        
        errors = []
        for pattern in error_patterns:
            matches = re.findall(pattern, error_output, re.MULTILINE)
            errors.extend(matches)
        
        if errors:
            return "\n".join(errors[:5])  # Limit to first 5 errors
        return error_output[:500]  # Fallback to first 500 chars
    
    def _extract_cuda_runtime_error(self, error_str: str) -> str:
        """Extract meaningful CUDA runtime error information"""
        cuda_error_mapping = {
            "illegal memory access": "Accessing memory outside allocated bounds - check array indices and pointer arithmetic",
            "invalid configuration argument": "Invalid kernel launch configuration - check grid/block dimensions",
            "out of memory": "GPU out of memory - reduce batch size or optimize memory usage",
            "no kernel image": "No kernel image available - ensure kernel is compiled for target GPU architecture",
            "unspecified launch failure": "Kernel launch failed - check for infinite loops, stack overflow, or assertion failures",
        }
        
        for error_key, explanation in cuda_error_mapping.items():
            if error_key in error_str.lower():
                return f"{error_str}\nExplanation: {explanation}"
        
        return error_str
    
    def _get_model_args(self, reference_model: Any) -> List:
        """Extract initialization arguments from reference model"""
        # This would need to be implemented based on the specific model architectures
        # For now, return empty list
        return []
    
    def _measure_speedup(
        self, 
        model_new: Any, 
        reference_model: Any,
        test_inputs: List[torch.Tensor],
        num_runs: int = 100
    ) -> float:
        """Measure speedup of new model vs reference"""
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model_new(*test_inputs[0])
                _ = reference_model(*test_inputs[0])
        
        # Measure new model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model_new(*test_inputs[0])
        torch.cuda.synchronize()
        new_time = time.time() - start
        
        # Measure reference model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = reference_model(*test_inputs[0])
        torch.cuda.synchronize()
        ref_time = time.time() - start
        
    def _verify_speedup(
        self,
        model_new: Any,
        reference_model: Any,
        test_inputs: List[torch.Tensor],
        num_runs: int = 1000
    ) -> float:
        """Verify speedup with more rigorous testing"""
        # Test with different input sizes to catch cheating
        varied_inputs = []
        for inp in test_inputs:
            # Create variations in size
            if isinstance(inp, tuple):
                varied_inputs.append(inp)
                # Add smaller and larger variants
                small_inp = tuple(t[:t.shape[0]//2] if t.shape[0] > 1 else t for t in inp)
                varied_inputs.append(small_inp)
            else:
                varied_inputs.append((inp,))
        
        speedups = []
        for test_inp in varied_inputs:
            try:
                # Ensure outputs still match
                with torch.no_grad():
                    out_new = model_new(*test_inp)
                    out_ref = reference_model(*test_inp)
                    if not torch.allclose(out_new, out_ref, rtol=1e-3, atol=1e-3):
                        return 0.0  # Cheating detected
                
                # Measure speedup for this input
                speedup = self._measure_speedup(model_new, reference_model, [test_inp], num_runs=100)
                speedups.append(speedup)
            except:
                return 0.0
        
        # Return median speedup (more robust against outliers)
        return np.median(speedups) if speedups else 0.0

class KernelBenchDataset:
    """Wrapper for KernelBench tasks"""
    
    def __init__(self, kernelbench_path: str, levels: List[int] = [1, 2]):
        self.kernelbench_path = kernelbench_path
        self.levels = levels
        self.tasks = self._load_tasks()
    
    def _load_tasks(self) -> List[Dict]:
        """Load tasks from KernelBench"""
        tasks = []
        # This would load actual tasks from KernelBench
        # For now, create example tasks
        example_tasks = [
            {
                "name": "matrix_multiply",
                "level": 1,
                "prompt": self._get_matrix_multiply_prompt(),
                "reference_model": self._get_matrix_multiply_reference(),
                "test_inputs": self._get_matrix_multiply_test_inputs()
            },
            {
                "name": "layer_norm",
                "level": 1,
                "prompt": self._get_layer_norm_prompt(),
                "reference_model": self._get_layer_norm_reference(),
                "test_inputs": self._get_layer_norm_test_inputs()
            }
        ]
        return example_tasks
    
    def _get_matrix_multiply_prompt(self) -> str:
        return """You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, M, N, K):
        super(Model, self).__init__()
        self.M = M
        self.N = N
        self.K = K
        
    def forward(self, A, B):
        # A is M x K, B is K x N
        return torch.matmul(A, B)

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100. 
Use techniques like:
- Shared memory for tile-based computation
- Coalesced memory access
- Warp-level primitives
- Tensor cores if applicable

Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew."""

    def _get_layer_norm_prompt(self) -> str:
        return """You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100.
Focus on:
- Efficient reduction using warp shuffle instructions
- Fused computation of mean, variance, and normalization
- Vectorized memory access
- Minimal global memory transactions

Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew."""

    def _get_matrix_multiply_reference(self):
        class Model(nn.Module):
            def __init__(self, M, N, K):
                super().__init__()
                self.M, self.N, self.K = M, N, K
            def forward(self, A, B):
                return torch.matmul(A, B)
        return Model(512, 512, 512)
    
    def _get_layer_norm_reference(self):
        return nn.LayerNorm((768,))
    
    def _get_matrix_multiply_test_inputs(self):
        return [
            (torch.randn(512, 512, device='cuda'), torch.randn(512, 512, device='cuda'))
            for _ in range(3)
        ]
    
    def _get_layer_norm_test_inputs(self):
        return [
            (torch.randn(32, 768, device='cuda'),)
            for _ in range(3)
        ]

class CUDAKernelRLTrainer:
    """Multi-turn RL trainer for CUDA kernel generation"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        learning_rate: float = 2e-6,
        gamma: float = 0.4,
        max_refinement_steps: int = 8,
        parallel_trajectories: int = 16,
        device: str = "cuda",
        max_prompt_length: int = 8192,
        max_response_length: int = 16384
    ):
        self.device = device
        self.gamma = gamma
        self.max_refinement_steps = max_refinement_steps
        self.parallel_trajectories = parallel_trajectories
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        self.kernel_executor = CUDAKernelExecutor()
        
    def generate_kernel(self, prompt: str, temperature: float = 0.8) -> Tuple[str, str]:
        """Generate CUDA kernel given a prompt"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_response_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract kernel code and summary
        kernel_code, summary = self._extract_kernel_and_summary(generated)
        return kernel_code, summary
    
    def _extract_kernel_and_summary(self, text: str) -> Tuple[str, str]:
        """Extract kernel code and brief summary from generated text"""
        # Extract summary (first few lines before code)
        lines = text.split('\n')
        summary_lines = []
        code_start = -1
        
        for i, line in enumerate(lines):
            if 'class ModelNew' in line or '```' in line:
                code_start = i
                break
            if line.strip() and not line.startswith('#'):
                summary_lines.append(line.strip())
        
        summary = ' '.join(summary_lines[:2])  # Max 2 sentences
        
        # Extract code starting from class ModelNew
        if code_start >= 0:
            code = '\n'.join(lines[code_start:])
        else:
            code = text
            
        return code, summary
    
    def create_refinement_prompt(
        self, 
        original_prompt: str, 
        previous_samples: List[KernelSample]
    ) -> str:
        """Create prompt for refinement step without chain of thought"""
        prompt = f"{original_prompt}\n\n"
        prompt += "Previous attempts:\n\n"
        
        # Include previous kernels and feedback (no CoT to avoid context explosion)
        for i, sample in enumerate(previous_samples[-3:]):  # Only last 3 attempts
            prompt += f"## Attempt {i+1}:\n"
            if sample.summary:
                prompt += f"Approach: {sample.summary}\n"
            prompt += f"Result: {sample.feedback}\n"
            
            # Add specific error context for failed attempts
            if not sample.compilation_success:
                prompt += "Status: Compilation Failed\n"
                # Extract specific error lines if present
                if "error:" in sample.feedback.lower():
                    prompt += "Key errors to fix:\n"
                    error_lines = [line for line in sample.feedback.split('\n') if 'error:' in line.lower()][:3]
                    for error in error_lines:
                        prompt += f"  - {error.strip()}\n"
                        
            elif not sample.is_correct:
                prompt += "Status: Runtime Error or Incorrect Output\n"
                # Add hints based on error type
                if "CUDA error" in sample.feedback:
                    prompt += "Hint: Check memory access patterns and thread indexing\n"
                elif "Max absolute diff" in sample.feedback:
                    prompt += "Hint: Algorithm produces wrong results - verify your computation logic\n"
                elif "Runtime error" in sample.feedback:
                    prompt += "Hint: Check for null pointers, array bounds, and synchronization\n"
                    
            else:
                prompt += f"Status: Correct! Current speedup: {sample.speedup:.2f}x\n"
            
            prompt += "\n"
        
        # Add refinement instruction based on last result
        last_sample = previous_samples[-1]
        
        if not last_sample.compilation_success:
            prompt += """Fix the compilation errors above. Focus on:
1. Correct CUDA syntax (__global__, __device__, etc.)
2. Proper kernel launch syntax (<<<blocks, threads>>>)
3. Include all necessary headers and declarations
4. Ensure ModelNew class is properly defined

First provide a brief summary (max 2 sentences) of your fix approach, then the corrected code."""
            
        elif not last_sample.is_correct:
            if "CUDA error" in last_sample.feedback:
                prompt += """Fix the CUDA runtime errors. Common causes:
1. Out-of-bounds memory access - check your indexing: idx < size
2. Shared memory bank conflicts - use padding if needed
3. Race conditions - ensure proper __syncthreads() usage
4. Null pointer access - initialize all pointers

First provide a brief summary (max 2 sentences) of your fix approach, then the corrected code."""
                
            else:
                prompt += """Fix the incorrect output. Debugging steps:
1. Verify your algorithm matches the reference implementation
2. Check for race conditions in parallel execution
3. Ensure proper reduction operations if used
4. Verify floating point precision handling

First provide a brief summary (max 2 sentences) of your fix approach, then the corrected code."""
                
        else:  # Kernel is correct, optimize further
            current_speedup = last_sample.speedup
            if current_speedup < 2.0:
                prompt += f"""The kernel is correct but slow ({current_speedup:.2f}x). Major optimizations needed:
- Use shared memory to reduce global memory access
- Implement tile-based algorithms for better cache usage
- Use warp shuffle instructions (__shfl_xor_sync) for reductions
- Consider memory coalescing patterns
- Use vectorized loads (float4) where possible"""
                
            elif current_speedup < 5.0:
                prompt += f"""Good progress ({current_speedup:.2f}x speedup). Further optimizations:
- Optimize bank conflicts in shared memory access
- Use warp-level primitives for faster reductions
- Implement loop unrolling for better instruction throughput
- Consider using texture memory for read-only data
- Tune block and grid dimensions for your GPU"""
                
            else:
                prompt += f"""Excellent speedup ({current_speedup:.2f}x)! Push further with advanced techniques:
- Use tensor cores if applicable (wmma operations)
- Implement persistent kernels to reduce launch overhead
- Try cooperative groups for flexible synchronization
- Optimize register usage to increase occupancy
- Consider multi-kernel fusion opportunities"""
                
            prompt += "\n\nFirst provide a brief summary (max 2 sentences) of your optimization approach, then the improved code."
            
        return prompt
    
    def generate_trajectory(
        self, 
        task: Dict
    ) -> KernelTrajectory:
        """Generate a complete trajectory with multiple refinement steps"""
        samples = []
        best_speedup = 0.0
        
        for step in range(self.max_refinement_steps):
            if step == 0:
                # Initial generation
                prompt = task["prompt"]
            else:
                # Refinement step
                prompt = self.create_refinement_prompt(task["prompt"], samples)
            
            # Generate kernel
            kernel_code, summary = self.generate_kernel(prompt)
            
            # Evaluate kernel
            compilation_success, is_correct, feedback, speedup = \
                self.kernel_executor.compile_and_test_kernel(
                    kernel_code,
                    task["reference_model"],
                    task["test_inputs"]
                )
            
            # Calculate reward
            reward = 0.0
            if compilation_success and not feedback.startswith("Reward hacking"):
                reward += 0.1  # Compilation reward
            if is_correct:
                reward += 0.3  # Correctness reward
                reward += speedup  # Performance reward
                best_speedup = max(best_speedup, speedup)
            
            # Penalize reward hacking attempts
            if "reward hacking" in feedback.lower():
                reward = -1.0  # Strong negative reward for hacking attempts
            
            sample = KernelSample(
                prompt=prompt,
                kernel_code=kernel_code,
                feedback=feedback,
                reward=reward,
                is_correct=is_correct,
                speedup=speedup,
                compilation_success=compilation_success,
                summary=summary
            )
            samples.append(sample)
            
            # Early stopping if we achieve very high speedup
            if is_correct and speedup > 10.0:
                break
        
        # Calculate discounted rewards
        for i in range(len(samples)):
            discounted_reward = sum(
                self.gamma ** j * samples[i + j].reward 
                for j in range(len(samples) - i)
            )
            samples[i].reward = discounted_reward
        
        total_reward = sum(s.reward for s in samples)
        
        return KernelTrajectory(
            initial_prompt=task["prompt"],
            task_name=task["name"],
            samples=samples,
            total_reward=total_reward,
            best_speedup=best_speedup
        )
    
    def compute_grpo_advantages(self, trajectories: List[KernelTrajectory]) -> List[float]:
        """Compute advantages using Group Relative Policy Optimization"""
        # Group trajectories by task
        task_groups = {}
        for traj in trajectories:
            if traj.task_name not in task_groups:
                task_groups[traj.task_name] = []
            task_groups[traj.task_name].append(traj)
        
        advantages = []
        for traj in trajectories:
            # Normalize within task group
            group = task_groups[traj.task_name]
            group_rewards = [t.total_reward for t in group]
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards) + 1e-8
            
            advantage = (traj.total_reward - mean_reward) / std_reward
            advantages.append(advantage)
            
        return advantages
    
    def train_step(self, tasks: List[Dict]):
        """Single training step with multiple trajectories per task"""
        all_trajectories = []
        
        # Generate parallel trajectories for each task
        print("Generating trajectories...")
        for task in tasks:
            task_trajectories = []
            for i in range(self.parallel_trajectories):
                print(f"  Task: {task['name']}, Trajectory {i+1}/{self.parallel_trajectories}")
                trajectory = self.generate_trajectory(task)
                task_trajectories.append(trajectory)
                
                # Log best result
                if trajectory.best_speedup > 0:
                    print(f"    Best speedup: {trajectory.best_speedup:.2f}x")
                    
            all_trajectories.extend(task_trajectories)
        
        # Compute advantages using GRPO
        advantages = self.compute_grpo_advantages(all_trajectories)
        
        # Training phase
        print("Training on generated trajectories...")
        total_loss = 0
        num_samples = 0
        self.model.train()
        
        for trajectory, advantage in zip(all_trajectories, advantages):
            for sample in trajectory.samples:
                # Skip if kernel generation failed completely
                if not sample.kernel_code:
                    continue
                    
                # Prepare inputs
                inputs = self.tokenizer(
                    sample.prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_length
                ).to(self.device)
                
                # Prepare targets (the generated kernel code)
                target_text = f"{sample.summary}\n{sample.kernel_code}" if sample.summary else sample.kernel_code
                targets = self.tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_response_length
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=targets['input_ids']
                )
                
                # Compute policy gradient loss with advantage and reward
                loss = outputs.loss * advantage * sample.reward
                
                # Backward pass
                loss.backward()
                total_loss += loss.item()
                num_samples += 1
        
        # Gradient clipping (aggressive clipping as in the paper)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        return avg_loss
    
    def train(
        self, 
        dataset: KernelBenchDataset,
        num_epochs: int = 100,
        tasks_per_batch: int = 8,
        save_freq: int = 10
    ):
        """Main training loop"""
        tasks = dataset.tasks
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle tasks
            np.random.shuffle(tasks)
            
            # Create batches
            for i in range(0, len(tasks), tasks_per_batch):
                batch_tasks = tasks[i:i+tasks_per_batch]
                
                print(f"\nBatch {i//tasks_per_batch + 1}, Tasks: {[t['name'] for t in batch_tasks]}")
                loss = self.train_step(batch_tasks)
                
                epoch_loss += loss
                num_batches += 1
                
                print(f"Batch loss: {loss:.4f}")
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f"cuda_kernel_checkpoint_epoch_{epoch+1}.pt")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

# Example usage
if __name__ == "__main__":
    # Initialize dataset
    dataset = KernelBenchDataset(kernelbench_path="/path/to/kernelbench")
    
    # Initialize trainer
    trainer = CUDAKernelRLTrainer(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",  # Start with smaller model
        learning_rate=2e-6,
        gamma=0.4,
        max_refinement_steps=4,  # Fewer steps for initial testing
        parallel_trajectories=4,  # Fewer trajectories for initial testing
        device="cuda"
    )
    
    # Train model
    trainer.train(
        dataset=dataset,
        num_epochs=50,
        tasks_per_batch=2,
        save_freq=10
    )
,  # Just returning input
            r'out\s*=\s*x\s*;?\s*
    
    def compile_and_test_kernel(
        self, 
        kernel_code: str, 
        reference_model: Any,
        test_inputs: List[torch.Tensor]
    ) -> Tuple[bool, bool, str, float]:
        """
        Compile and test CUDA kernel
        Returns: (compilation_success, is_correct, feedback, speedup)
        """
        try:
            # Create temporary file for the kernel code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Add necessary imports
                full_code = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import time

""" + kernel_code
                
                f.write(full_code)
                f.flush()
                
                # Try to compile and load the kernel
                try:
                    # Dynamic import
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("kernel_module", f.name)
                    kernel_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(kernel_module)
                    
                    ModelNew = kernel_module.ModelNew
                    model_new = ModelNew(*self._get_model_args(reference_model))
                    
                except Exception as e:
                    return False, False, f"Compilation error: {str(e)}", 0.0
                
                # Test correctness
                model_new.eval()
                reference_model.eval()
                
                with torch.no_grad():
                    for test_input in test_inputs:
                        try:
                            output_new = model_new(*test_input)
                            output_ref = reference_model(*test_input)
                            
                            if not torch.allclose(output_new, output_ref, rtol=1e-3, atol=1e-3):
                                return True, False, "Incorrect output: results don't match reference", 0.0
                        except Exception as e:
                            return True, False, f"Runtime error: {str(e)}", 0.0
                
                # Measure performance
                speedup = self._measure_speedup(model_new, reference_model, test_inputs)
                
                feedback = f"Kernel compiled and passed all tests! Speedup: {speedup:.2f}x"
                return True, True, feedback, speedup
                
        except Exception as e:
            return False, False, f"Unexpected error: {str(e)}", 0.0
        finally:
            if 'f' in locals():
                os.unlink(f.name)
    
    def _get_model_args(self, reference_model: Any) -> List:
        """Extract initialization arguments from reference model"""
        # This would need to be implemented based on the specific model architectures
        # For now, return empty list
        return []
    
    def _measure_speedup(
        self, 
        model_new: Any, 
        reference_model: Any,
        test_inputs: List[torch.Tensor],
        num_runs: int = 100
    ) -> float:
        """Measure speedup of new model vs reference"""
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model_new(*test_inputs[0])
                _ = reference_model(*test_inputs[0])
        
        # Measure new model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model_new(*test_inputs[0])
        torch.cuda.synchronize()
        new_time = time.time() - start
        
        # Measure reference model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = reference_model(*test_inputs[0])
        torch.cuda.synchronize()
        ref_time = time.time() - start
        
        return ref_time / new_time
    
    def _verify_speedup(
        self,
        model_new: Any,
        reference_model: Any,
        test_inputs: List[torch.Tensor],
        num_runs: int = 1000
    ) -> float:
        """Verify speedup with more rigorous testing"""
        # Test with different input sizes to catch cheating
        varied_inputs = []
        for inp in test_inputs:
            # Create variations in size
            if isinstance(inp, tuple):
                varied_inputs.append(inp)
                # Add smaller and larger variants
                small_inp = tuple(t[:t.shape[0]//2] if t.shape[0] > 1 else t for t in inp)
                varied_inputs.append(small_inp)
            else:
                varied_inputs.append((inp,))
        
        speedups = []
        for test_inp in varied_inputs:
            try:
                # Ensure outputs still match
                with torch.no_grad():
                    out_new = model_new(*test_inp)
                    out_ref = reference_model(*test_inp)
                    if not torch.allclose(out_new, out_ref, rtol=1e-3, atol=1e-3):
                        return 0.0  # Cheating detected
                
                # Measure speedup for this input
                speedup = self._measure_speedup(model_new, reference_model, [test_inp], num_runs=100)
                speedups.append(speedup)
            except:
                return 0.0
        
        # Return median speedup (more robust against outliers)
        return np.median(speedups) if speedups else 0.0

class KernelBenchDataset:
    """Wrapper for KernelBench tasks"""
    
    def __init__(self, kernelbench_path: str, levels: List[int] = [1, 2]):
        self.kernelbench_path = kernelbench_path
        self.levels = levels
        self.tasks = self._load_tasks()
    
    def _load_tasks(self) -> List[Dict]:
        """Load tasks from KernelBench"""
        tasks = []
        # This would load actual tasks from KernelBench
        # For now, create example tasks
        example_tasks = [
            {
                "name": "matrix_multiply",
                "level": 1,
                "prompt": self._get_matrix_multiply_prompt(),
                "reference_model": self._get_matrix_multiply_reference(),
                "test_inputs": self._get_matrix_multiply_test_inputs()
            },
            {
                "name": "layer_norm",
                "level": 1,
                "prompt": self._get_layer_norm_prompt(),
                "reference_model": self._get_layer_norm_reference(),
                "test_inputs": self._get_layer_norm_test_inputs()
            }
        ]
        return example_tasks
    
    def _get_matrix_multiply_prompt(self) -> str:
        return """You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, M, N, K):
        super(Model, self).__init__()
        self.M = M
        self.N = N
        self.K = K
        
    def forward(self, A, B):
        # A is M x K, B is K x N
        return torch.matmul(A, B)

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100. 
Use techniques like:
- Shared memory for tile-based computation
- Coalesced memory access
- Warp-level primitives
- Tensor cores if applicable

Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew."""

    def _get_layer_norm_prompt(self) -> str:
        return """You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100.
Focus on:
- Efficient reduction using warp shuffle instructions
- Fused computation of mean, variance, and normalization
- Vectorized memory access
- Minimal global memory transactions

Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew."""

    def _get_matrix_multiply_reference(self):
        class Model(nn.Module):
            def __init__(self, M, N, K):
                super().__init__()
                self.M, self.N, self.K = M, N, K
            def forward(self, A, B):
                return torch.matmul(A, B)
        return Model(512, 512, 512)
    
    def _get_layer_norm_reference(self):
        return nn.LayerNorm((768,))
    
    def _get_matrix_multiply_test_inputs(self):
        return [
            (torch.randn(512, 512, device='cuda'), torch.randn(512, 512, device='cuda'))
            for _ in range(3)
        ]
    
    def _get_layer_norm_test_inputs(self):
        return [
            (torch.randn(32, 768, device='cuda'),)
            for _ in range(3)
        ]

class CUDAKernelRLTrainer:
    """Multi-turn RL trainer for CUDA kernel generation"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        learning_rate: float = 2e-6,
        gamma: float = 0.4,
        max_refinement_steps: int = 8,
        parallel_trajectories: int = 16,
        device: str = "cuda",
        max_prompt_length: int = 8192,
        max_response_length: int = 16384
    ):
        self.device = device
        self.gamma = gamma
        self.max_refinement_steps = max_refinement_steps
        self.parallel_trajectories = parallel_trajectories
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        self.kernel_executor = CUDAKernelExecutor()
        
    def generate_kernel(self, prompt: str, temperature: float = 0.8) -> Tuple[str, str]:
        """Generate CUDA kernel given a prompt"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_response_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract kernel code and summary
        kernel_code, summary = self._extract_kernel_and_summary(generated)
        return kernel_code, summary
    
    def _extract_kernel_and_summary(self, text: str) -> Tuple[str, str]:
        """Extract kernel code and brief summary from generated text"""
        # Extract summary (first few lines before code)
        lines = text.split('\n')
        summary_lines = []
        code_start = -1
        
        for i, line in enumerate(lines):
            if 'class ModelNew' in line or '```' in line:
                code_start = i
                break
            if line.strip() and not line.startswith('#'):
                summary_lines.append(line.strip())
        
        summary = ' '.join(summary_lines[:2])  # Max 2 sentences
        
        # Extract code starting from class ModelNew
        if code_start >= 0:
            code = '\n'.join(lines[code_start:])
        else:
            code = text
            
        return code, summary
    
    def create_refinement_prompt(
        self, 
        original_prompt: str, 
        previous_samples: List[KernelSample]
    ) -> str:
        """Create prompt for refinement step without chain of thought"""
        prompt = f"{original_prompt}\n\n"
        prompt += "Previous attempts:\n\n"
        
        # Include previous kernels and feedback (no CoT to avoid context explosion)
        for i, sample in enumerate(previous_samples[-3:]):  # Only last 3 attempts
            prompt += f"## Attempt {i+1}:\n"
            if sample.summary:
                prompt += f"Approach: {sample.summary}\n"
            prompt += f"Result: {sample.feedback}\n"
            if sample.is_correct:
                prompt += f"Current speedup: {sample.speedup:.2f}x\n"
            prompt += "\n"
        
        # Add refinement instruction based on last result
        last_sample = previous_samples[-1]
        if last_sample.compilation_success and last_sample.is_correct:
            prompt += f"""The kernel is correct with {last_sample.speedup:.2f}x speedup. 
Optimize further using more aggressive techniques like:
- Warp shuffle instructions (__shfl_xor_sync)
- Tensor cores for matrix operations
- Better memory access patterns
- Loop unrolling
- Reduced bank conflicts

First provide a brief summary (max 2 sentences) of your optimization approach, then the code."""
        elif last_sample.compilation_success:
            prompt += "Fix the correctness issues in the kernel. Brief summary first, then code."
        else:
            prompt += "Fix the compilation errors. Brief summary first, then code."
            
        return prompt
    
    def generate_trajectory(
        self, 
        task: Dict
    ) -> KernelTrajectory:
        """Generate a complete trajectory with multiple refinement steps"""
        samples = []
        best_speedup = 0.0
        
        for step in range(self.max_refinement_steps):
            if step == 0:
                # Initial generation
                prompt = task["prompt"]
            else:
                # Refinement step
                prompt = self.create_refinement_prompt(task["prompt"], samples)
            
            # Generate kernel
            kernel_code, summary = self.generate_kernel(prompt)
            
            # Evaluate kernel
            compilation_success, is_correct, feedback, speedup = \
                self.kernel_executor.compile_and_test_kernel(
                    kernel_code,
                    task["reference_model"],
                    task["test_inputs"]
                )
            
            # Calculate reward
            reward = 0.0
            if compilation_success:
                reward += 0.1  # Compilation reward
            if is_correct:
                reward += 0.3  # Correctness reward
                reward += speedup  # Performance reward
                best_speedup = max(best_speedup, speedup)
            
            sample = KernelSample(
                prompt=prompt,
                kernel_code=kernel_code,
                feedback=feedback,
                reward=reward,
                is_correct=is_correct,
                speedup=speedup,
                compilation_success=compilation_success,
                summary=summary
            )
            samples.append(sample)
            
            # Early stopping if we achieve very high speedup
            if is_correct and speedup > 10.0:
                break
        
        # Calculate discounted rewards
        for i in range(len(samples)):
            discounted_reward = sum(
                self.gamma ** j * samples[i + j].reward 
                for j in range(len(samples) - i)
            )
            samples[i].reward = discounted_reward
        
        total_reward = sum(s.reward for s in samples)
        
        return KernelTrajectory(
            initial_prompt=task["prompt"],
            task_name=task["name"],
            samples=samples,
            total_reward=total_reward,
            best_speedup=best_speedup
        )
    
    def compute_grpo_advantages(self, trajectories: List[KernelTrajectory]) -> List[float]:
        """Compute advantages using Group Relative Policy Optimization"""
        # Group trajectories by task
        task_groups = {}
        for traj in trajectories:
            if traj.task_name not in task_groups:
                task_groups[traj.task_name] = []
            task_groups[traj.task_name].append(traj)
        
        advantages = []
        for traj in trajectories:
            # Normalize within task group
            group = task_groups[traj.task_name]
            group_rewards = [t.total_reward for t in group]
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards) + 1e-8
            
            advantage = (traj.total_reward - mean_reward) / std_reward
            advantages.append(advantage)
            
        return advantages
    
    def train_step(self, tasks: List[Dict]):
        """Single training step with multiple trajectories per task"""
        all_trajectories = []
        
        # Generate parallel trajectories for each task
        print("Generating trajectories...")
        for task in tasks:
            task_trajectories = []
            for i in range(self.parallel_trajectories):
                print(f"  Task: {task['name']}, Trajectory {i+1}/{self.parallel_trajectories}")
                trajectory = self.generate_trajectory(task)
                task_trajectories.append(trajectory)
                
                # Log best result
                if trajectory.best_speedup > 0:
                    print(f"    Best speedup: {trajectory.best_speedup:.2f}x")
                    
            all_trajectories.extend(task_trajectories)
        
        # Compute advantages using GRPO
        advantages = self.compute_grpo_advantages(all_trajectories)
        
        # Training phase
        print("Training on generated trajectories...")
        total_loss = 0
        num_samples = 0
        self.model.train()
        
        for trajectory, advantage in zip(all_trajectories, advantages):
            for sample in trajectory.samples:
                # Skip if kernel generation failed completely
                if not sample.kernel_code:
                    continue
                    
                # Prepare inputs
                inputs = self.tokenizer(
                    sample.prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_length
                ).to(self.device)
                
                # Prepare targets (the generated kernel code)
                target_text = f"{sample.summary}\n{sample.kernel_code}" if sample.summary else sample.kernel_code
                targets = self.tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_response_length
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=targets['input_ids']
                )
                
                # Compute policy gradient loss with advantage and reward
                loss = outputs.loss * advantage * sample.reward
                
                # Backward pass
                loss.backward()
                total_loss += loss.item()
                num_samples += 1
        
        # Gradient clipping (aggressive clipping as in the paper)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        return avg_loss
    
    def train(
        self, 
        dataset: KernelBenchDataset,
        num_epochs: int = 100,
        tasks_per_batch: int = 8,
        save_freq: int = 10
    ):
        """Main training loop"""
        tasks = dataset.tasks
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle tasks
            np.random.shuffle(tasks)
            
            # Create batches
            for i in range(0, len(tasks), tasks_per_batch):
                batch_tasks = tasks[i:i+tasks_per_batch]
                
                print(f"\nBatch {i//tasks_per_batch + 1}, Tasks: {[t['name'] for t in batch_tasks]}")
                loss = self.train_step(batch_tasks)
                
                epoch_loss += loss
                num_batches += 1
                
                print(f"Batch loss: {loss:.4f}")
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f"cuda_kernel_checkpoint_epoch_{epoch+1}.pt")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

# Example usage
if __name__ == "__main__":
    # Initialize dataset
    dataset = KernelBenchDataset(kernelbench_path="/path/to/kernelbench")
    
    # Initialize trainer
    trainer = CUDAKernelRLTrainer(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",  # Start with smaller model
        learning_rate=2e-6,
        gamma=0.4,
        max_refinement_steps=4,  # Fewer steps for initial testing
        parallel_trajectories=4,  # Fewer trajectories for initial testing
        device="cuda"
    )
    
    # Train model
    trainer.train(
        dataset=dataset,
        num_epochs=50,
        tasks_per_batch=2,
        save_freq=10
    )
,  # Copying input to output
            r'memcpy\(.*?\)',  # Just memcpy without computation
        ]
        for pattern in dummy_patterns:
            if re.search(pattern, code, re.MULTILINE):
                return True, "Dummy implementation detected (no actual computation)"
        
        # Verify CUDA kernel presence
        if not self._has_cuda_kernel(code):
            return True, "No CUDA kernel implementation found"
        
        # Check for minimal kernel that doesn't do real work
        kernel_body = self._extract_kernel_body(code)
        if kernel_body and len(kernel_body.strip().split('\n')) < 5:
            return True, "CUDA kernel too simple (likely not implementing the algorithm)"
        
        return False, ""
    
    def _has_cuda_kernel(self, code: str) -> bool:
        """Check if code contains actual CUDA kernel"""
        cuda_indicators = [
            r'__global__\s+void',  # CUDA kernel declaration
            r'<<<.*>>>',  # CUDA kernel launch
            r'blockIdx', r'threadIdx', r'blockDim', r'gridDim',  # CUDA built-ins
            r'__shared__',  # Shared memory
            r'__syncthreads',  # Synchronization
            r'cuda_sources\s*=',  # load_inline cuda_sources
        ]
        
        cuda_count = sum(1 for pattern in cuda_indicators 
                        if re.search(pattern, code))
        
        # Need at least 3 CUDA indicators for a real kernel
        return cuda_count >= 3
    
    def _extract_kernel_body(self, code: str) -> str:
        """Extract the main CUDA kernel body"""
        kernel_pattern = r'__global__\s+void\s+\w+\s*\([^)]*\)\s*{(.*?)}'
        match = re.search(kernel_pattern, code, re.DOTALL)
        return match.group(1) if match else ""
    
    def compile_and_test_kernel(
        self, 
        kernel_code: str, 
        reference_model: Any,
        test_inputs: List[torch.Tensor]
    ) -> Tuple[bool, bool, str, float]:
        """
        Compile and test CUDA kernel
        Returns: (compilation_success, is_correct, feedback, speedup)
        """
        try:
            # Create temporary file for the kernel code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Add necessary imports
                full_code = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import time

""" + kernel_code
                
                f.write(full_code)
                f.flush()
                
                # Try to compile and load the kernel
                try:
                    # Dynamic import
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("kernel_module", f.name)
                    kernel_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(kernel_module)
                    
                    ModelNew = kernel_module.ModelNew
                    model_new = ModelNew(*self._get_model_args(reference_model))
                    
                except Exception as e:
                    return False, False, f"Compilation error: {str(e)}", 0.0
                
                # Test correctness
                model_new.eval()
                reference_model.eval()
                
                with torch.no_grad():
                    for test_input in test_inputs:
                        try:
                            output_new = model_new(*test_input)
                            output_ref = reference_model(*test_input)
                            
                            if not torch.allclose(output_new, output_ref, rtol=1e-3, atol=1e-3):
                                return True, False, "Incorrect output: results don't match reference", 0.0
                        except Exception as e:
                            return True, False, f"Runtime error: {str(e)}", 0.0
                
                # Measure performance
                speedup = self._measure_speedup(model_new, reference_model, test_inputs)
                
                feedback = f"Kernel compiled and passed all tests! Speedup: {speedup:.2f}x"
                return True, True, feedback, speedup
                
        except Exception as e:
            return False, False, f"Unexpected error: {str(e)}", 0.0
        finally:
            if 'f' in locals():
                os.unlink(f.name)
    
    def _get_model_args(self, reference_model: Any) -> List:
        """Extract initialization arguments from reference model"""
        # This would need to be implemented based on the specific model architectures
        # For now, return empty list
        return []
    
    def _measure_speedup(
        self, 
        model_new: Any, 
        reference_model: Any,
        test_inputs: List[torch.Tensor],
        num_runs: int = 100
    ) -> float:
        """Measure speedup of new model vs reference"""
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model_new(*test_inputs[0])
                _ = reference_model(*test_inputs[0])
        
        # Measure new model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model_new(*test_inputs[0])
        torch.cuda.synchronize()
        new_time = time.time() - start
        
        # Measure reference model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = reference_model(*test_inputs[0])
        torch.cuda.synchronize()
        ref_time = time.time() - start
        
        return ref_time / new_time

class KernelBenchDataset:
    """Wrapper for KernelBench tasks"""
    
    def __init__(self, kernelbench_path: str, levels: List[int] = [1, 2]):
        self.kernelbench_path = kernelbench_path
        self.levels = levels
        self.tasks = self._load_tasks()
    
    def _load_tasks(self) -> List[Dict]:
        """Load tasks from KernelBench"""
        tasks = []
        # This would load actual tasks from KernelBench
        # For now, create example tasks
        example_tasks = [
            {
                "name": "matrix_multiply",
                "level": 1,
                "prompt": self._get_matrix_multiply_prompt(),
                "reference_model": self._get_matrix_multiply_reference(),
                "test_inputs": self._get_matrix_multiply_test_inputs()
            },
            {
                "name": "layer_norm",
                "level": 1,
                "prompt": self._get_layer_norm_prompt(),
                "reference_model": self._get_layer_norm_reference(),
                "test_inputs": self._get_layer_norm_test_inputs()
            }
        ]
        return example_tasks
    
    def _get_matrix_multiply_prompt(self) -> str:
        return """You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, M, N, K):
        super(Model, self).__init__()
        self.M = M
        self.N = N
        self.K = K
        
    def forward(self, A, B):
        # A is M x K, B is K x N
        return torch.matmul(A, B)

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100. 
Use techniques like:
- Shared memory for tile-based computation
- Coalesced memory access
- Warp-level primitives
- Tensor cores if applicable

Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew."""

    def _get_layer_norm_prompt(self) -> str:
        return """You are given the following architecture:

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100.
Focus on:
- Efficient reduction using warp shuffle instructions
- Fused computation of mean, variance, and normalization
- Vectorized memory access
- Minimal global memory transactions

Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew."""

    def _get_matrix_multiply_reference(self):
        class Model(nn.Module):
            def __init__(self, M, N, K):
                super().__init__()
                self.M, self.N, self.K = M, N, K
            def forward(self, A, B):
                return torch.matmul(A, B)
        return Model(512, 512, 512)
    
    def _get_layer_norm_reference(self):
        return nn.LayerNorm((768,))
    
    def _get_matrix_multiply_test_inputs(self):
        return [
            (torch.randn(512, 512, device='cuda'), torch.randn(512, 512, device='cuda'))
            for _ in range(3)
        ]
    
    def _get_layer_norm_test_inputs(self):
        return [
            (torch.randn(32, 768, device='cuda'),)
            for _ in range(3)
        ]

class CUDAKernelRLTrainer:
    """Multi-turn RL trainer for CUDA kernel generation"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        learning_rate: float = 2e-6,
        gamma: float = 0.4,
        max_refinement_steps: int = 8,
        parallel_trajectories: int = 16,
        device: str = "cuda",
        max_prompt_length: int = 8192,
        max_response_length: int = 16384
    ):
        self.device = device
        self.gamma = gamma
        self.max_refinement_steps = max_refinement_steps
        self.parallel_trajectories = parallel_trajectories
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        self.kernel_executor = CUDAKernelExecutor()
        
    def generate_kernel(self, prompt: str, temperature: float = 0.8) -> Tuple[str, str]:
        """Generate CUDA kernel given a prompt"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_response_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract kernel code and summary
        kernel_code, summary = self._extract_kernel_and_summary(generated)
        return kernel_code, summary
    
    def _extract_kernel_and_summary(self, text: str) -> Tuple[str, str]:
        """Extract kernel code and brief summary from generated text"""
        # Extract summary (first few lines before code)
        lines = text.split('\n')
        summary_lines = []
        code_start = -1
        
        for i, line in enumerate(lines):
            if 'class ModelNew' in line or '```' in line:
                code_start = i
                break
            if line.strip() and not line.startswith('#'):
                summary_lines.append(line.strip())
        
        summary = ' '.join(summary_lines[:2])  # Max 2 sentences
        
        # Extract code starting from class ModelNew
        if code_start >= 0:
            code = '\n'.join(lines[code_start:])
        else:
            code = text
            
        return code, summary
    
    def create_refinement_prompt(
        self, 
        original_prompt: str, 
        previous_samples: List[KernelSample]
    ) -> str:
        """Create prompt for refinement step without chain of thought"""
        prompt = f"{original_prompt}\n\n"
        prompt += "Previous attempts:\n\n"
        
        # Include previous kernels and feedback (no CoT to avoid context explosion)
        for i, sample in enumerate(previous_samples[-3:]):  # Only last 3 attempts
            prompt += f"## Attempt {i+1}:\n"
            if sample.summary:
                prompt += f"Approach: {sample.summary}\n"
            prompt += f"Result: {sample.feedback}\n"
            if sample.is_correct:
                prompt += f"Current speedup: {sample.speedup:.2f}x\n"
            prompt += "\n"
        
        # Add refinement instruction based on last result
        last_sample = previous_samples[-1]
        if last_sample.compilation_success and last_sample.is_correct:
            prompt += f"""The kernel is correct with {last_sample.speedup:.2f}x speedup. 
Optimize further using more aggressive techniques like:
- Warp shuffle instructions (__shfl_xor_sync)
- Tensor cores for matrix operations
- Better memory access patterns
- Loop unrolling
- Reduced bank conflicts

First provide a brief summary (max 2 sentences) of your optimization approach, then the code."""
        elif last_sample.compilation_success:
            prompt += "Fix the correctness issues in the kernel. Brief summary first, then code."
        else:
            prompt += "Fix the compilation errors. Brief summary first, then code."
            
        return prompt
    
    def generate_trajectory(
        self, 
        task: Dict
    ) -> KernelTrajectory:
        """Generate a complete trajectory with multiple refinement steps"""
        samples = []
        best_speedup = 0.0
        
        for step in range(self.max_refinement_steps):
            if step == 0:
                # Initial generation
                prompt = task["prompt"]
            else:
                # Refinement step
                prompt = self.create_refinement_prompt(task["prompt"], samples)
            
            # Generate kernel
            kernel_code, summary = self.generate_kernel(prompt)
            
            # Evaluate kernel
            compilation_success, is_correct, feedback, speedup = \
                self.kernel_executor.compile_and_test_kernel(
                    kernel_code,
                    task["reference_model"],
                    task["test_inputs"]
                )
            
            # Calculate reward
            reward = 0.0
            if compilation_success:
                reward += 0.1  # Compilation reward
            if is_correct:
                reward += 0.3  # Correctness reward
                reward += speedup  # Performance reward
                best_speedup = max(best_speedup, speedup)
            
            sample = KernelSample(
                prompt=prompt,
                kernel_code=kernel_code,
                feedback=feedback,
                reward=reward,
                is_correct=is_correct,
                speedup=speedup,
                compilation_success=compilation_success,
                summary=summary
            )
            samples.append(sample)
            
            # Early stopping if we achieve very high speedup
            if is_correct and speedup > 10.0:
                break
        
        # Calculate discounted rewards
        for i in range(len(samples)):
            discounted_reward = sum(
                self.gamma ** j * samples[i + j].reward 
                for j in range(len(samples) - i)
            )
            samples[i].reward = discounted_reward
        
        total_reward = sum(s.reward for s in samples)
        
        return KernelTrajectory(
            initial_prompt=task["prompt"],
            task_name=task["name"],
            samples=samples,
            total_reward=total_reward,
            best_speedup=best_speedup
        )
    
    def compute_grpo_advantages(self, trajectories: List[KernelTrajectory]) -> List[float]:
        """Compute advantages using Group Relative Policy Optimization"""
        # Group trajectories by task
        task_groups = {}
        for traj in trajectories:
            if traj.task_name not in task_groups:
                task_groups[traj.task_name] = []
            task_groups[traj.task_name].append(traj)
        
        advantages = []
        for traj in trajectories:
            # Normalize within task group
            group = task_groups[traj.task_name]
            group_rewards = [t.total_reward for t in group]
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards) + 1e-8
            
            advantage = (traj.total_reward - mean_reward) / std_reward
            advantages.append(advantage)
            
        return advantages
    
    def train_step(self, tasks: List[Dict]):
        """Single training step with multiple trajectories per task"""
        all_trajectories = []
        
        # Generate parallel trajectories for each task
        print("Generating trajectories...")
        for task in tasks:
            task_trajectories = []
            for i in range(self.parallel_trajectories):
                print(f"  Task: {task['name']}, Trajectory {i+1}/{self.parallel_trajectories}")
                trajectory = self.generate_trajectory(task)
                task_trajectories.append(trajectory)
                
                # Log best result
                if trajectory.best_speedup > 0:
                    print(f"    Best speedup: {trajectory.best_speedup:.2f}x")
                    
            all_trajectories.extend(task_trajectories)
        
        # Compute advantages using GRPO
        advantages = self.compute_grpo_advantages(all_trajectories)
        
        # Training phase
        print("Training on generated trajectories...")
        total_loss = 0
        num_samples = 0
        self.model.train()
        
        for trajectory, advantage in zip(all_trajectories, advantages):
            for sample in trajectory.samples:
                # Skip if kernel generation failed completely
                if not sample.kernel_code:
                    continue
                    
                # Prepare inputs
                inputs = self.tokenizer(
                    sample.prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_length
                ).to(self.device)
                
                # Prepare targets (the generated kernel code)
                target_text = f"{sample.summary}\n{sample.kernel_code}" if sample.summary else sample.kernel_code
                targets = self.tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_response_length
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=targets['input_ids']
                )
                
                # Compute policy gradient loss with advantage and reward
                loss = outputs.loss * advantage * sample.reward
                
                # Backward pass
                loss.backward()
                total_loss += loss.item()
                num_samples += 1
        
        # Gradient clipping (aggressive clipping as in the paper)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        return avg_loss
    
    def train(
        self, 
        dataset: KernelBenchDataset,
        num_epochs: int = 100,
        tasks_per_batch: int = 8,
        save_freq: int = 10
    ):
        """Main training loop"""
        tasks = dataset.tasks
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle tasks
            np.random.shuffle(tasks)
            
            # Create batches
            for i in range(0, len(tasks), tasks_per_batch):
                batch_tasks = tasks[i:i+tasks_per_batch]
                
                print(f"\nBatch {i//tasks_per_batch + 1}, Tasks: {[t['name'] for t in batch_tasks]}")
                loss = self.train_step(batch_tasks)
                
                epoch_loss += loss
                num_batches += 1
                
                print(f"Batch loss: {loss:.4f}")
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f"cuda_kernel_checkpoint_epoch_{epoch+1}.pt")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

# Example usage
if __name__ == "__main__":
    # Initialize dataset
    dataset = KernelBenchDataset(kernelbench_path="/path/to/kernelbench")
    
    # Initialize trainer
    trainer = CUDAKernelRLTrainer(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",  # Start with smaller model
        learning_rate=2e-6,
        gamma=0.4,
        max_refinement_steps=4,  # Fewer steps for initial testing
        parallel_trajectories=4,  # Fewer trajectories for initial testing
        device="cuda"
    )
    
    # Train model
    trainer.train(
        dataset=dataset,
        num_epochs=50,
        tasks_per_batch=2,
        save_freq=10
    )
