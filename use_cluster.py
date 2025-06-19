#!/usr/bin/env python3
"""
Distributed training script for CUDA kernel generation with multi-turn RL
Supports multi-node training with DeepSpeed and SLURM
"""

import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
from deepspeed import comm as dist_comm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import wandb
from typing import Dict, List, Optional
import socket
import subprocess

# Import the main training modules
from cuda_kernel_rl_trainer import (
    CUDAKernelRLTrainer, 
    KernelBenchDataset,
    KernelTrajectory,
    KernelSample
)

class DistributedCUDAKernelTrainer:
    """Distributed wrapper for CUDA kernel training"""
    
    def __init__(
        self,
        args: argparse.Namespace,
        deepspeed_config: Dict
    ):
        self.args = args
        self.deepspeed_config = deepspeed_config
        
        # Setup distributed environment
        self.setup_distributed()
        
        # Initialize wandb on rank 0
        if self.is_main_process:
            wandb.init(
                project="cuda-kernel-rl",
                name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args)
            )
        
        # Setup logging
        self.setup_logging()
        
    def setup_distributed(self):
        """Setup distributed training environment"""
        if 'SLURM_PROCID' in os.environ:
            # SLURM environment
            self.rank = int(os.environ['SLURM_PROCID'])
            self.world_size = int(os.environ['SLURM_NTASKS'])
            self.local_rank = int(os.environ['SLURM_LOCALID'])
            
            # Get master address from SLURM
            node_list = os.environ['SLURM_NODELIST']
            master_node = subprocess.check_output(
                f'scontrol show hostname {node_list} | head -n1', 
                shell=True
            ).decode().strip()
            
            os.environ['MASTER_ADDR'] = master_node
            os.environ['MASTER_PORT'] = str(self.args.master_port)
            
        else:
            # Standard distributed setup
            self.rank = int(os.environ.get('RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            rank=self.rank,
            world_size=self.world_size
        )
        
        # Set device
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        self.is_main_process = self.rank == 0
        
    def setup_logging(self):
        """Setup distributed logging"""
        log_level = logging.INFO if self.is_main_process else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format=f'[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/rank_{self.rank}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_model_and_optimizer(self):
        """Create model with DeepSpeed"""
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Model configuration
        model_config = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        
        # Create model with DeepSpeed
        with deepspeed.zero.Init(enabled=self.args.zero_stage == 3):
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                **model_config
            )
        
        # Initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=self.deepspeed_config
        )
        
        return model_engine, optimizer, tokenizer
    
    def distributed_generate_trajectories(
        self,
        trainer: CUDAKernelRLTrainer,
        tasks: List[Dict],
        trajectories_per_task: int
    ) -> List[KernelTrajectory]:
        """Generate trajectories in a distributed manner"""
        # Distribute tasks across ranks
        tasks_per_rank = len(tasks) // self.world_size
        rank_tasks = tasks[
            self.rank * tasks_per_rank : (self.rank + 1) * tasks_per_rank
        ]
        
        # Handle remainder tasks
        remainder = len(tasks) % self.world_size
        if self.rank < remainder:
            rank_tasks.append(tasks[self.world_size * tasks_per_rank + self.rank])
        
        # Generate trajectories for assigned tasks
        local_trajectories = []
        for task in rank_tasks:
            self.logger.info(f"Generating trajectories for task: {task['name']}")
            
            for i in range(trajectories_per_task):
                trajectory = trainer.generate_trajectory(task)
                local_trajectories.append(trajectory)
                
                if trajectory.best_speedup > 0:
                    self.logger.info(
                        f"Task {task['name']}, Trajectory {i+1}: "
                        f"Best speedup {trajectory.best_speedup:.2f}x"
                    )
        
        # Gather trajectories from all ranks
        all_trajectories = self.gather_trajectories(local_trajectories)
        
        return all_trajectories
    
    def gather_trajectories(
        self, 
        local_trajectories: List[KernelTrajectory]
    ) -> List[KernelTrajectory]:
        """Gather trajectories from all ranks"""
        # Serialize trajectories
        serialized_local = [
            self.serialize_trajectory(traj) for traj in local_trajectories
        ]
        
        # Gather sizes first
        local_size = torch.tensor([len(serialized_local)], device=self.device)
        sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(sizes, local_size)
        
        # Prepare gather lists
        max_size = max(s.item() for s in sizes)
        padded_trajectories = serialized_local + [''] * (max_size - len(serialized_local))
        
        # All-gather trajectories
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, padded_trajectories)
        
        # Deserialize and combine
        all_trajectories = []
        for rank_trajs, size in zip(gathered, sizes):
            for i in range(size.item()):
                if rank_trajs[i]:  # Skip padding
                    traj = self.deserialize_trajectory(rank_trajs[i])
                    all_trajectories.append(traj)
        
        return all_trajectories
    
    def serialize_trajectory(self, trajectory: KernelTrajectory) -> str:
        """Serialize trajectory for distribution"""
        data = {
            'initial_prompt': trajectory.initial_prompt,
            'task_name': trajectory.task_name,
            'total_reward': trajectory.total_reward,
            'best_speedup': trajectory.best_speedup,
            'samples': [
                {
                    'prompt': s.prompt,
                    'kernel_code': s.kernel_code,
                    'feedback': s.feedback,
                    'reward': s.reward,
                    'is_correct': s.is_correct,
                    'speedup': s.speedup,
                    'compilation_success': s.compilation_success,
                    'summary': s.summary
                }
                for s in trajectory.samples
            ]
        }
        return json.dumps(data)
    
    def deserialize_trajectory(self, data_str: str) -> KernelTrajectory:
        """Deserialize trajectory"""
        data = json.loads(data_str)
        samples = [
            KernelSample(**sample_data) for sample_data in data['samples']
        ]
        return KernelTrajectory(
            initial_prompt=data['initial_prompt'],
            task_name=data['task_name'],
            samples=samples,
            total_reward=data['total_reward'],
            best_speedup=data['best_speedup']
        )
    
    def train(self):
        """Main distributed training loop"""
        # Create model and optimizer with DeepSpeed
        model_engine, optimizer, tokenizer = self.create_model_and_optimizer()
        
        # Initialize dataset
        dataset = KernelBenchDataset(
            kernelbench_path=self.args.kernelbench_path,
            levels=self.args.levels
        )
        
        # Create trainer instance (without its own optimizer)
        trainer = CUDAKernelRLTrainer(
            model_name=self.args.model_name,
            device=self.device,
            max_refinement_steps=self.args.max_refinement_steps,
            parallel_trajectories=1  # Generate one at a time, distribute across ranks
        )
        
        # Replace model and tokenizer with distributed versions
        trainer.model = model_engine
        trainer.tokenizer = tokenizer
        trainer.optimizer = None  # Will use DeepSpeed optimizer
        
        # Training loop
        for epoch in range(self.args.num_epochs):
            if self.is_main_process:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Epoch {epoch+1}/{self.args.num_epochs}")
                self.logger.info(f"{'='*50}")
            
            # Shuffle tasks with same seed across ranks
            tasks = dataset.tasks.copy()
            rng = np.random.RandomState(epoch)
            rng.shuffle(tasks)
            
            # Process batches
            for batch_idx in range(0, len(tasks), self.args.tasks_per_batch):
                batch_tasks = tasks[batch_idx:batch_idx + self.args.tasks_per_batch]
                
                if self.is_main_process:
                    self.logger.info(
                        f"\nBatch {batch_idx//self.args.tasks_per_batch + 1}, "
                        f"Tasks: {[t['name'] for t in batch_tasks]}"
                    )
                
                # Generate trajectories in distributed manner
                all_trajectories = self.distributed_generate_trajectories(
                    trainer, 
                    batch_tasks,
                    self.args.trajectories_per_task
                )
                
                # Compute advantages
                advantages = trainer.compute_grpo_advantages(all_trajectories)
                
                # Training step with gradient accumulation
                total_loss = self.distributed_train_step(
                    model_engine,
                    tokenizer,
                    all_trajectories,
                    advantages
                )
                
                # Logging
                if self.is_main_process:
                    avg_speedup = np.mean([t.best_speedup for t in all_trajectories])
                    successful = sum(1 for t in all_trajectories if t.best_speedup > 0)
                    
                    self.logger.info(f"Batch loss: {total_loss:.4f}")
                    self.logger.info(f"Average best speedup: {avg_speedup:.2f}x")
                    self.logger.info(f"Successful kernels: {successful}/{len(all_trajectories)}")
                    
                    wandb.log({
                        'epoch': epoch + 1,
                        'batch': batch_idx // self.args.tasks_per_batch + 1,
                        'loss': total_loss,
                        'avg_speedup': avg_speedup,
                        'success_rate': successful / len(all_trajectories)
                    })
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(model_engine, epoch + 1)
    
    def distributed_train_step(
        self,
        model_engine,
        tokenizer,
        trajectories: List[KernelTrajectory],
        advantages: List[float]
    ) -> float:
        """Distributed training step"""
        model_engine.train()
        total_loss = 0
        num_samples = 0
        
        # Process trajectories
        for trajectory, advantage in zip(trajectories, advantages):
            for sample in trajectory.samples:
                if not sample.kernel_code:
                    continue
                
                # Prepare inputs
                inputs = tokenizer(
                    sample.prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.args.max_prompt_length
                ).to(self.device)
                
                # Prepare targets
                target_text = f"{sample.summary}\n{sample.kernel_code}" if sample.summary else sample.kernel_code
                targets = tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.args.max_response_length
                ).to(self.device)
                
                # Forward pass
                outputs = model_engine(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=targets['input_ids']
                )
                
                # Compute loss with advantage and reward
                loss = outputs.loss * advantage * sample.reward
                
                # Backward pass with DeepSpeed
                model_engine.backward(loss)
                
                total_loss += loss.item()
                num_samples += 1
                
                # Step optimizer if gradient accumulation is complete
                if num_samples % self.args.gradient_accumulation_steps == 0:
                    model_engine.step()
        
        # Final optimizer step if needed
        if num_samples % self.args.gradient_accumulation_steps != 0:
            model_engine.step()
        
        # Reduce loss across ranks
        avg_loss = total_loss / max(num_samples, 1)
        avg_loss_tensor = torch.tensor([avg_loss], device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        
        return avg_loss_tensor.item()
    
    def save_checkpoint(self, model_engine, epoch: int):
        """Save distributed checkpoint"""
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint_epoch_{epoch}"
        
        # DeepSpeed saves model and optimizer state
        model_engine.save_checkpoint(
            checkpoint_dir,
            tag=f"epoch_{epoch}",
            client_state={'epoch': epoch}
        )
        
        if self.is_main_process:
            self.logger.info(f"Checkpoint saved to {checkpoint_dir}")

def get_deepspeed_config(args):
    """Generate DeepSpeed configuration"""
    config = {
        "train_batch_size": args.tasks_per_batch * args.trajectories_per_task * args.world_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": args.grad_clip_norm,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": args.zero_stage,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps
            }
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }
    }
    
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed CUDA Kernel Training")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--kernelbench_path", type=str, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2])
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--tasks_per_batch", type=int, default=8)
    parser.add_argument("--trajectories_per_task", type=int, default=16)
    parser.add_argument("--max_refinement_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--grad_clip_norm", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    # Model configuration
    parser.add_argument("--max_prompt_length", type=int, default=8192)
    parser.add_argument("--max_response_length", type=int, default=16384)
    
    # DeepSpeed arguments
    parser.add_argument("--zero_stage", type=int, default=3)
    parser.add_argument("--world_size", type=int, default=None)
    
    # Infrastructure arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--exp_name", type=str, default="cuda_kernel_rl")
    
    args = parser.parse_args()
    
    # Set world size if not specified
    if args.world_size is None:
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    return args

def main():
    args = parse_args()
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Generate DeepSpeed config
    deepspeed_config = get_deepspeed_config(args)
    
    # Initialize distributed trainer
    trainer = DistributedCUDAKernelTrainer(args, deepspeed_config)
    
    # Start training
    trainer.train()
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
