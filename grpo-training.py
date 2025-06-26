"""
GRPO Training Loop for CUDA Optimization
Implements Group Relative Policy Optimization with curriculum learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os
from collections import defaultdict
import wandb
from tqdm import tqdm
import random

from cuda_rl_system import (
    KernelState, OptimizationTrajectory, CurriculumBuffer,
    KernelEvaluator, RewardCalculator
)
from gemm_specialist import GEMMSpecialist

class CUDAKernelDataset(Dataset):
    """Dataset of CUDA kernels for optimization"""
    
    def __init__(self, kernels_file: str, problem_type: str = "gemm"):
        """
        Load kernels from file or KernelBench
        Format: JSON with {"problem_id": str, "code": str, "type": str}
        """
        self.problem_type = problem_type
        self.kernels = []
        
        if os.path.exists(kernels_file):
            with open(kernels_file, 'r') as f:
                self.kernels = json.load(f)
        else:
            # Generate synthetic kernels for testing
            self.kernels = self._generate_synthetic_kernels()
    
    def _generate_synthetic_kernels(self) -> List[Dict]:
        """Generate synthetic GEMM kernels with variations"""
        kernels = []
        
        # Base naive GEMM template
        base_template = """
__global__ void gemm_{variant}(float* A, float* B, float* C, int M, int N, int K) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {{
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {{
            sum += A[{access_A}] * B[{access_B}];
        }}
        C[row * N + col] = sum;
    }}
}}
"""
        
        # Variations in memory access patterns
        variations = [
            {"variant": "row_major", "access_A": "row * K + k", "access_B": "k * N + col"},
            {"variant": "col_major", "access_A": "k * M + row", "access_B": "col * K + k"},
            {"variant": "strided", "access_A": "row * K + k", "access_B": "(k * N + col) * 2 % (K * N)"},
        ]
        
        for i, var in enumerate(variations):
            kernel = {
                "problem_id": f"synthetic_gemm_{i:03d}",
                "code": base_template.format(**var),
                "type": "gemm",
                "difficulty": "easy" if i < 10 else "medium"
            }
            kernels.append(kernel)
        
        return kernels
    
    def __len__(self):
        return len(self.kernels)
    
    def __getitem__(self, idx):
        return self.kernels[idx]

class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer for CUDA optimization
    Based on ether0's approach with curriculum learning
    """
    
    def __init__(self,
                 specialist: GEMMSpecialist,
                 dataset: CUDAKernelDataset,
                 group_size: int = 4,
                 group_batch_size: int = 128,
                 learning_rate: float = 1e-6,
                 kl_penalty: float = 0.005,
                 epsilon_cur: float = 0.5,
                 device: str = "cuda"):
        
        self.specialist = specialist
        self.dataset = dataset
        self.group_size = group_size
        self.group_batch_size = group_batch_size
        self.learning_rate = learning_rate
        self.kl_penalty = kl_penalty
        self.epsilon_cur = epsilon_cur
        self.device = device
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.specialist.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Reference policy (frozen copy)
        self.reference_model = self._create_reference_model()
        
        # Training statistics
        self.training_stats = defaultdict(list)
        
    def _create_reference_model(self):
        """Create a frozen copy of the model as reference"""
        reference_model = type(self.specialist.model).from_pretrained(
            self.specialist.model_name,
            torch_dtype=self.specialist.model.dtype,
            device_map="auto"
        )
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        return reference_model
    
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute normalized advantages for a group"""
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) if np.std(rewards) > 0 else 1.0
        
        advantages = (rewards - mean_reward) / std_reward
        return advantages.tolist()
    
    def compute_kl_penalty(self, 
                          logprobs: torch.Tensor,
                          ref_logprobs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence penalty between policy and reference"""
        # Approximate KL divergence (from ether0 paper)
        kl = ref_logprobs - logprobs
        return self.kl_penalty * kl.mean()
    
    def train_step(self, problems: List[Dict]) -> Dict[str, float]:
        """Single GRPO training step"""
        
        # Collect rollouts for each problem
        all_trajectories = []
        all_rewards = []
        all_advantages = []
        problem_groups = []
        
        for problem in problems:
            # Generate group_size trajectories for this problem
            trajectories = []
            rewards = []
            
            for _ in range(self.group_size):
                trajectory = self.specialist.optimize_kernel(
                    initial_code=problem['code'],
                    problem_id=problem['problem_id'],
                    use_search=False,  # No search during training
                    max_turns=3
                )
                
                trajectories.append(trajectory)
                rewards.append(trajectory.total_reward)
            
            # Compute advantages for this group
            advantages = self.compute_advantages(rewards)
            
            # Store for training
            all_trajectories.extend(trajectories)
            all_rewards.extend(rewards)
            all_advantages.extend(advantages)
            problem_groups.append({
                'problem_id': problem['problem_id'],
                'rewards': rewards,
                'has_variance': np.var(rewards) > 0.01
            })
        
        # Update curriculum buffer
        for group in problem_groups:
            if group['has_variance']:
                self.specialist.curriculum_buffer.add_group(
                    group['problem_id'],
                    group['rewards']
                )
        
        # Compute policy gradients
        total_loss = 0.0
        total_pg_loss = 0.0
        total_kl_loss = 0.0
        
        self.optimizer.zero_grad()
        
        for i, (trajectory, advantage) in enumerate(zip(all_trajectories, all_advantages)):
            # Skip if advantage is zero (no learning signal)
            if abs(advantage) < 1e-6:
                continue
            
            # Compute loss for each state transition in trajectory
            for j in range(1, len(trajectory.states)):
                prev_state = trajectory.states[j-1]
                curr_state = trajectory.states[j]
                
                # Get action (the optimization applied)
                action_prompt = self._create_action_prompt(prev_state, curr_state)
                
                # Compute log probabilities
                with torch.no_grad():
                    ref_logprobs = self._compute_logprobs(
                        self.reference_model,
                        action_prompt,
                        curr_state.code
                    )
                
                logprobs = self._compute_logprobs(
                    self.specialist.model,
                    action_prompt,
                    curr_state.code
                )
                
                # Policy gradient loss (PPO-style clipping)
                ratio = torch.exp(logprobs - ref_logprobs)
                pg_loss1 = -advantage * ratio
                pg_loss2 = -advantage * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # KL penalty
                kl_loss = self.compute_kl_penalty(logprobs, ref_logprobs)
                
                # Total loss
                loss = pg_loss + kl_loss
                loss.backward()
                
                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_kl_loss += kl_loss.item()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.specialist.model.parameters(), 1.0)
        
        # Update model
        self.optimizer.step()
        
        # Compute statistics
        stats = {
            'total_loss': total_loss / len(all_trajectories),
            'pg_loss': total_pg_loss / len(all_trajectories),
            'kl_loss': total_kl_loss / len(all_trajectories),
            'mean_reward': np.mean(all_rewards),
            'reward_variance': np.var(all_rewards),
            'non_trivial_fraction': sum(1 for g in problem_groups if g['has_variance']) / len(problem_groups)
        }
        
        return stats
    
    def _create_action_prompt(self, prev_state: KernelState, curr_state: KernelState) -> str:
        """Create prompt representing the action taken"""
        return f"""Previous kernel:
```cuda
{prev_state.code}
```

Apply {curr_state.optimization_applied} optimization to get:
```cuda
{curr_state.code}
```
"""
    
    def _compute_logprobs(self, model, prompt: str, target_code: str) -> torch.Tensor:
        """Compute log probabilities of generating target code given prompt"""
        
        # Tokenize prompt and target
        prompt_tokens = self.specialist.tokenizer(prompt, return_tensors="pt").to(self.device)
        target_tokens = self.specialist.tokenizer(target_code, return_tensors="pt").to(self.device)
        
        # Concatenate prompt and target
        input_ids = torch.cat([prompt_tokens.input_ids, target_tokens.input_ids], dim=1)
        
        # Get model outputs
        with torch.no_grad() if model == self.reference_model else torch.enable_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
        
        # Compute log probabilities for target tokens
        target_start = prompt_tokens.input_ids.shape[1]
        target_logits = logits[:, target_start-1:-1]  # Shift for next token prediction
        target_ids = input_ids[:, target_start:]
        
        # Gather log probabilities
        log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return selected_log_probs.sum()
    
    def train(self, 
              num_steps: int = 1000,
              eval_interval: int = 50,
              save_interval: int = 100,
              use_wandb: bool = True):
        """Main training loop"""
        
        if use_wandb:
            wandb.init(
                project="cuda-optimization-rl",
                config={
                    "group_size": self.group_size,
                    "group_batch_size": self.group_batch_size,
                    "learning_rate": self.learning_rate,
                    "kl_penalty": self.kl_penalty,
                    "epsilon_cur": self.epsilon_cur
                }
            )
        
        # Training loop
        for step in tqdm(range(num_steps), desc="Training"):
            # Sample batch with curriculum
            batch = self._sample_batch()
            
            # Training step
            stats = self.train_step(batch)
            
            # Log statistics
            self.training_stats['step'].append(step)
            for key, value in stats.items():
                self.training_stats[key].append(value)
            
            if use_wandb:
                wandb.log(stats, step=step)
            
            # Evaluation
            if step % eval_interval == 0:
                eval_stats = self.evaluate()
                print(f"Step {step}: {eval_stats}")
                
                if use_wandb:
                    wandb.log(eval_stats, step=step)
            
            # Save checkpoint
            if step % save_interval == 0:
                self.save_checkpoint(step)
            
            # Update reference policy periodically (like ether0)
            if step % 256 == 0 and step > 0:
                print("Updating reference policy...")
                self.reference_model = self._create_reference_model()
    
    def _sample_batch(self) -> List[Dict]:
        """Sample batch using curriculum buffer"""
        batch_problems = []
        
        # Sample from curriculum buffer
        curriculum_ids = self.specialist.curriculum_buffer.sample(
            self.group_batch_size,
            self.epsilon_cur
        )
        
        # Get problems from curriculum
        for problem_id in curriculum_ids:
            # Find problem in dataset
            for kernel in self.dataset:
                if kernel['problem_id'] == problem_id:
                    batch_problems.append(kernel)
                    break
        
        # Fill rest with random samples
        while len(batch_problems) < self.group_batch_size:
            idx = random.randint(0, len(self.dataset) - 1)
            batch_problems.append(self.dataset[idx])
        
        return batch_problems[:self.group_batch_size]
    
    def evaluate(self, num_eval: int = 10) -> Dict[str, float]:
        """Evaluate current policy"""
        eval_rewards = []
        eval_improvements = []
        
        # Sample random evaluation problems
        eval_indices = random.sample(range(len(self.dataset)), min(num_eval, len(self.dataset)))
        
        for idx in eval_indices:
            problem = self.dataset[idx]
            
            # Run optimization with search
            trajectory = self.specialist.optimize_kernel(
                initial_code=problem['code'],
                problem_id=problem['problem_id'],
                use_search=True,
                max_turns=4
            )
            
            eval_rewards.append(trajectory.total_reward)
            
            # Calculate improvement
            initial_gflops = trajectory.states[0].performance_metrics.get('gflops', 0)
            final_gflops = trajectory.current_state.performance_metrics.get('gflops', 0)
            improvement = final_gflops / max(initial_gflops, 1.0)
            eval_improvements.append(improvement)
        
        return {
            'eval_mean_reward': np.mean(eval_rewards),
            'eval_mean_improvement': np.mean(eval_improvements),
            'eval_max_improvement': np.max(eval_improvements) if eval_improvements else 0
        }
    
    def save_checkpoint(self, step: int):
        """Save training checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.specialist.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': dict(self.training_stats),
            'curriculum_buffer': {
                'buffer': list(self.specialist.curriculum_buffer.buffer),
                'stats': self.specialist.curriculum_buffer.problem_stats
            }
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/gemm_specialist_step_{step}.pt')
        print(f"Saved checkpoint at step {step}")

# ===== Training Script =====

def train_gemm_specialist():
    """Main training function"""
    
    # Create dataset
    dataset = CUDAKernelDataset("kernels.json")  # Will generate synthetic if not exists
    print(f"Loaded {len(dataset)} kernels")
    
    # Create specialist
    from cuda_rl_system import BeamSearch
    specialist = GEMMSpecialist(
        search_algorithm=BeamSearch(beam_width=4)  # Smaller beam for training
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        specialist=specialist,
        dataset=dataset,
        group_size=4,
        group_batch_size=32,  # Smaller for testing
        learning_rate=1e-6,
        kl_penalty=0.005,
        epsilon_cur=0.5
    )
    
    # Train
    trainer.train(
        num_steps=100,  # Small number for testing
        eval_interval=10,
        save_interval=20,
        use_wandb=False  # Set to True if you have wandb configured
    )
    
    # Save final model
    torch.save(
        specialist.model.state_dict(),
        "gemm_specialist_final.pt"
    )
    print("Training complete!")
    
    # Print final statistics
    print("\nFinal Training Statistics:")
    for key in ['mean_reward', 'reward_variance', 'non_trivial_fraction']:
        if key in trainer.training_stats:
            values = trainer.training_stats[key]
            print(f"{key}: {values[-1]:.4f} (initial: {values[0]:.4f})")

if __name__ == "__main__":
    train_gemm_specialist()
