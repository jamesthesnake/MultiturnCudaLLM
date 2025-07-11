#!/usr/bin/env python3
"""
Kevin-32B Inference Test Script
Tests the model with various CUDA kernel optimization tasks
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import argparse
import sys
import os
from typing import Optional
import re

# Test problems for Kevin-32B
TEST_PROBLEMS = {
    "matmul": {
        "name": "Matrix Multiplication",
        "code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, M=1024, N=1024, K=1024):
        super(Model, self).__init__()
        self.M = M
        self.N = N
        self.K = K
        
    def forward(self, A, B):
        # A is M x K, B is K x N
        return torch.matmul(A, B)""",
        "description": "Optimize matrix multiplication for GPU performance"
    },
    
    "layernorm": {
        "name": "Layer Normalization",
        "code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, normalized_shape=(768,)):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        
    def forward(self, x):
        return self.ln(x)""",
        "description": "Optimize LayerNorm with efficient reduction"
    },
    
    "conv2d": {
        "name": "2D Convolution",
        "code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv(x)""",
        "description": "Optimize 2D convolution operation"
    },
    
    "softmax": {
        "name": "Softmax",
        "code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim=-1):
        super(Model, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return torch.softmax(x, dim=self.dim)""",
        "description": "Optimize softmax with numerical stability"
    },
    
    "gelu": {
        "name": "GELU Activation",
        "code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x):
        return torch.nn.functional.gelu(x)""",
        "description": "Optimize GELU activation function"
    }
}


class KevinInferenceTester:
    def __init__(
        self,
        model_name: str = "cognition-ai/Kevin-32B",
        device: str = "auto",
        load_in_8bit: bool = False,
        use_auth_token: Optional[str] = None
    ):
        """Initialize Kevin-32B for testing"""
        print(f"🚀 Loading Kevin-32B from {model_name}...")
        print(f"   Device: {device}")
        print(f"   8-bit quantization: {load_in_8bit}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=use_auth_token,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Model loading configuration
            model_kwargs = {
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "use_auth_token": use_auth_token,
            }
            
            if device == "auto":
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = {"": device}
            
            if load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                
            # Try to use Flash Attention 2 if available
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("   Using Flash Attention 2 ✓")
            except:
                print("   Flash Attention 2 not available, using default attention")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            self.device = device
            print("✅ Model loaded successfully!\n")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you have enough RAM/VRAM")
            print("2. Try using --load-in-8bit flag")
            print("3. Check your HuggingFace token if the model is gated")
            sys.exit(1)
    
    def create_prompt(self, problem: dict) -> str:
        """Create a prompt in Kevin-32B's expected format"""
        prompt = f"""You are given the following architecture:

{problem['code']}

Replace pytorch operators in the given architecture with raw CUDA kernels, optimizing for performance on NVIDIA H100 (you can also optimize for other GPUs like A100, A40, or V100).

{problem['description']}

Use techniques like:
- Shared memory for tile-based computation
- Coalesced memory access
- Warp-level primitives (__shfl_xor_sync for reductions)
- Tensor cores if applicable (for GEMM operations)
- Vectorized loads (float4) where possible
- Bank conflict avoidance
- Loop unrolling

Use torch.utils.cpp_extension.load_inline and name your optimized output architecture ModelNew.

Your answer must be the complete new architecture (no testing code, no other code). Provide the full implementation that can be directly executed."""
        
        return prompt
    
    def generate_kernel(
        self,
        prompt: str,
        max_new_tokens: int = 8192,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> str:
        """Generate CUDA kernel using Kevin-32B"""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        
        # Move to device if not using auto
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        print("🤖 Generating optimized CUDA kernel...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"✅ Generation complete in {generation_time:.2f}s")
        
        return generated_text
    
    def extract_code(self, generated_text: str) -> Optional[str]:
        """Extract the ModelNew implementation from generated text"""
        # Try to find class ModelNew
        pattern = r'class ModelNew.*?(?=(?:class|def|\n\n# |\Z))'
        match = re.search(pattern, generated_text, re.DOTALL)
        
        if match:
            return match.group(0).strip()
        
        # Alternative: look for code blocks
        code_pattern = r'```(?:python|cuda)?\n(.*?)```'
        code_matches = re.findall(code_pattern, generated_text, re.DOTALL)
        
        for code in code_matches:
            if 'class ModelNew' in code:
                return code.strip()
        
        return None
    
    def test_problem(self, problem_key: str):
        """Test Kevin-32B on a specific problem"""
        if problem_key not in TEST_PROBLEMS:
            print(f"❌ Unknown problem: {problem_key}")
            print(f"Available problems: {list(TEST_PROBLEMS.keys())}")
            return
        
        problem = TEST_PROBLEMS[problem_key]
        
        print(f"{'='*60}")
        print(f"🎯 Testing: {problem['name']}")
        print(f"{'='*60}\n")
        
        # Show original code
        print("📋 Original PyTorch Code:")
        print("-" * 40)
        print(problem['code'])
        print("-" * 40)
        print()
        
        # Create prompt
        prompt = self.create_prompt(problem)
        
        # Generate kernel
        generated_text = self.generate_kernel(prompt)
        
        # Extract code
        kernel_code = self.extract_code(generated_text)
        
        print("\n📝 Generated Response:")
        print("-" * 40)
        
        if kernel_code:
            print(kernel_code)
            
            # Quick analysis
            print("\n📊 Quick Analysis:")
            if "__global__" in kernel_code:
                print("✓ Contains CUDA kernel declaration")
            if "__shared__" in kernel_code:
                print("✓ Uses shared memory")
            if "__syncthreads()" in kernel_code:
                print("✓ Uses thread synchronization")
            if "blockIdx" in kernel_code and "threadIdx" in kernel_code:
                print("✓ Uses CUDA thread indexing")
            if "__shfl_xor_sync" in kernel_code:
                print("✓ Uses warp shuffle operations")
            if "wmma" in kernel_code or "tensor_core" in kernel_code.lower():
                print("✓ Uses tensor cores")
                
        else:
            print("❌ Could not extract ModelNew implementation")
            print("\nFull generated text:")
            print(generated_text[:1000] + "..." if len(generated_text) > 1000 else generated_text)
        
        print("-" * 40)
        print()
    
    def test_all(self):
        """Test all available problems"""
        print(f"🧪 Testing Kevin-32B on {len(TEST_PROBLEMS)} problems\n")
        
        for problem_key in TEST_PROBLEMS:
            self.test_problem(problem_key)
            print("\n" + "="*60 + "\n")
    
    def interactive_mode(self):
        """Interactive mode for custom kernels"""
        print("🎮 Interactive Mode - Enter your PyTorch code")
        print("Type 'END' on a new line when done\n")
        
        while True:
            print("Enter PyTorch code (or 'quit' to exit):")
            lines = []
            while True:
                line = input()
                if line == 'END':
                    break
                if line == 'quit':
                    return
                lines.append(line)
            
            if not lines:
                continue
                
            pytorch_code = '\n'.join(lines)
            
            # Create custom problem
            custom_problem = {
                'code': pytorch_code,
                'description': 'Optimize this custom kernel for GPU performance'
            }
            
            prompt = self.create_prompt(custom_problem)
            generated_text = self.generate_kernel(prompt)
            kernel_code = self.extract_code(generated_text)
            
            if kernel_code:
                print("\n🚀 Optimized CUDA Kernel:")
                print("-" * 40)
                print(kernel_code)
                print("-" * 40)
            else:
                print("\n❌ Could not generate valid kernel")
                print(generated_text[:500] + "...")
            
            print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test Kevin-32B CUDA kernel generation")
    parser.add_argument(
        "--model",
        type=str,
        default="cognition-ai/Kevin-32B",
        help="Model name or path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on (auto, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization (reduces memory usage)"
    )
    parser.add_argument(
        "--problem",
        type=str,
        choices=list(TEST_PROBLEMS.keys()) + ['all'],
        default='matmul',
        help="Problem to test"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode for custom kernels"
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        help="HuggingFace auth token if needed"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device != "auto" and args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("❌ CUDA is not available. Please use CPU or check your installation.")
            sys.exit(1)
        print(f"🎮 CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Initialize tester
    tester = KevinInferenceTester(
        model_name=args.model,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        use_auth_token=args.auth_token
    )
    
    # Run tests
    if args.interactive:
        tester.interactive_mode()
    elif args.problem == 'all':
        tester.test_all()
    else:
        tester.test_problem(args.problem)


if __name__ == "__main__":
    main()
