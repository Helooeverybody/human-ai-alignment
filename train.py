import torch 
from OmegaConf import DictConfig
import torch.nn as nn
from typing import Optional
import torch.distributed as dist 
import os 
import wandb
import trainers

def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str= 'nccl'):
    print(rank, "Initializing distributed training...")
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) # Assuming one GPU per process instead of put everything on the same GPU


def main_worker(
        rank: int, 
        world_size: int, 
        config: DictConfig,
        policy: nn.Module, 
        reference: Optional[nn.Module]
):
    if 'FSDP' in config.train.strategy:
        init_distributed(rank, world_size, port = config.fsdp_port)
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
    
    #---------TODO---------#
    if rank == 0 and config.wandb.enabled:
        pass

    TrainerClass = getattr(trainers, config.trainer)
    trainer  = TrainerClass()


    
    