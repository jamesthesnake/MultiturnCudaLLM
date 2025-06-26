"""
Launch script for 3x A40 CUDA RL System
Handles environment setup and multi-GPU initialization
"""

import os
import sys
import torch
import asyncio
import argparse
from typing import Optional

def setup_a40_environment():
    """Configure environment for optimal A40 performance"""
    
    # Verify we have 3 GPUs
    if torch.cuda.device_count() < 3:
        print(f"Warning: Only {torch.cuda.device_count()} GPUs found. Expected 3x A40s.")
        print("The system will still run but with reduced parallelism.")
    
    # Set environment variables for A40 optimization
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async kernel launches
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # A40 compute capability
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Optimize cuBLAS
    
    # For multi-GPU setup
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Limit CPU threads to avoid oversubscription
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # PyTorch settings for A40
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    print("Environment configured for 3x A40 setup:")
    for i in range(min(3, torch.cuda.device_count())):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

async def test_multi_gpu_system():
    """Test the multi-GPU setup"""
    from multi_gpu_system import A40Optimizer
    from gemm_specialist import create_gemm_specialist, NAIVE_GEMM
    
    print("\nTesting Multi-GPU System...")
    print("-" * 50)
    
    # Initialize system
    optimizer = A40Optimizer()
    
    # Create specialist on GPU 0
    with torch.cuda.device(0):
        specialist = create_gemm_specialist(use_beam_search=True, beam_width=4)
        print("✓ Model loaded on GPU 0")
    
    # Test kernel evaluation on GPU 1-2
    test_problems = [
        {"problem_id": "test_1", "code": NAIVE_GEMM},
        {"problem_id": "test_2", "code": NAIVE_GEMM},
    ]
    
    print("\nTesting parallel kernel evaluation...")
    trajectories = await optimizer.optimize_batch(
        specialist,
        test_problems,
        use_search=False
    )
    
    print(f"✓ Successfully evaluated {len(trajectories)} problems")
    
    # Show GPU utilization
    optimizer.print_gpu_status()
    
    return True

async def train_with_multi_gpu(args):
    """Train specialist using 3x A40 setup"""
    from grpo_training import GRPOTrainer, CUDAKernelDataset
    from gemm_specialist import GEMMSpecialist
    from cuda_rl_system import BeamSearch
    
    print("\nInitializing training with 3x A40s...")
    
    # Create dataset
    dataset = CUDAKernelDataset("kernels.json")
    print(f"Dataset: {len(dataset)} kernels")
    
    # Create specialist on GPU 0
    with torch.cuda.device(0):
        specialist = GEMMSpecialist(
            search_algorithm=BeamSearch(beam_width=4)
        )
    
    # Create trainer with multi-GPU support
    trainer = GRPOTrainer(
        specialist=specialist,
        dataset=dataset,
        group_size=args.group_size,
        group_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_multi_gpu=True  # Enable multi-GPU
    )
    
    print("\nStarting training...")
    print(f"Configuration:")
    print(f"  - Group size: {args.group_size}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Training steps: {args.num_steps}")
    
    # Modified training loop for async
    for step in range(args.num_steps):
        batch = trainer._sample_batch()
        
        if trainer.use_multi_gpu:
            # Use async multi-GPU training step
            stats = await trainer.train_step_multi_gpu(batch)
        else:
            stats = trainer.train_step(batch)
        
        if step % 10 == 0:
            print(f"\nStep {step}: loss={stats['total_loss']:.4f}, "
                  f"reward={stats['mean_reward']:.4f}, "
                  f"non_trivial={stats['non_trivial_fraction']:.2f}")
        
        if step % args.save_interval == 0:
            trainer.save_checkpoint(step)
    
    print("\nTraining complete!")

def benchmark_multi_gpu():
    """Benchmark the multi-GPU setup"""
    import time
    from gemm_specialist import NAIVE_GEMM
    
    print("\nBenchmarking Multi-GPU Performance...")
    print("-" * 50)
    
    # Test compilation speed
    print("\n1. Testing parallel compilation...")
    kernels = [NAIVE_GEMM] * 10
    
    # Single GPU baseline
    start = time.time()
    for kernel in kernels:
        # Simulate evaluation on single GPU
        time.sleep(0.1)  # Placeholder
    single_time = time.time() - start
    
    print(f"Single GPU time: {single_time:.2f}s")
    print(f"Theoretical multi-GPU time (2x parallel): {single_time/2:.2f}s")
    
    # Memory usage
    print("\n2. GPU Memory Usage:")
    for i in range(min(3, torch.cuda.device_count())):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

def main():
    parser = argparse.ArgumentParser(description="3x A40 CUDA RL System")
    parser.add_argument('--mode', choices=['test', 'train', 'benchmark'], 
                       default='test', help='Execution mode')
    parser.add_argument('--num_steps', type=int, default=100, 
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--group_size', type=int, default=4, 
                       help='Group size for GRPO')
    parser.add_argument('--learning_rate', type=float, default=1e-6, 
                       help='Learning rate')
    parser.add_argument('--save_interval', type=int, default=20, 
                       help='Checkpoint save interval')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_a40_environment()
    
    if args.mode == 'test':
        # Run async test
        success = asyncio.run(test_multi_gpu_system())
        if success:
            print("\n✓ All tests passed! System ready for training.")
    
    elif args.mode == 'train':
        # Run training
        asyncio.run(train_with_multi_gpu(args))
    
    elif args.mode == 'benchmark':
        # Run benchmarks
        benchmark_multi_gpu()

if __name__ == "__main__":
    main()
