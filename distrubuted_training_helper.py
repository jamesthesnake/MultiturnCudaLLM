import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from typing import List, Dict

class DistributedKernelTrainer:
    """Coordinates training across multiple GPUs"""
    
    def __init__(self, rank: int, world_size: int, config):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.setup_distributed()
        
    def setup_distributed(self):
        """Initialize distributed training"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
        
        # Set device
        torch.cuda.set_device(self.rank)
        self.device = torch.device(f'cuda:{self.rank}')
        
    def setup_model_ddp(self, model):
        """Wrap model in DDP"""
        model = model.to(self.device)
        return DDP(model, device_ids=[self.rank], output_device=self.rank)
    
    def coordinate_specialists(self):
        """Coordinate specialist training across GPUs"""
        specialist_assignments = {
            0: ["memory_management", "scheduling"],
            1: ["drivers", "filesystems"],
            2: ["networking", "security"]
        }
        
        my_specialties = specialist_assignments.get(self.rank, [])
        return my_specialties
    
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate metrics across all processes"""
        for key, value in metrics.items():
            tensor = torch.tensor(value).to(self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            metrics[key] = tensor.item()
        return metrics
