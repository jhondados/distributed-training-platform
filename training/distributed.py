"""Distributed training with PyTorch FSDP."""
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

def setup_fsdp_training(model, rank: int, world_size: int, model_layer_cls=None):
    """Configure FSDP for efficient distributed LLM training."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    auto_wrap_policy = None
    if model_layer_cls:
        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={model_layer_cls})
    model = model.cuda(rank)
    fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision, auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, cpu_offload=None,
        device_id=torch.cuda.current_device())
    return fsdp_model

class DistributedTrainer:
    def __init__(self, fsdp_model, optimizer, scheduler=None,
                 gradient_checkpointing: bool = True, max_grad_norm: float = 1.0):
        self.model = fsdp_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)  # disabled with bf16
        if gradient_checkpointing:
            self.model.enable_input_require_grads()

    def train_step(self, batch: dict) -> dict:
        self.optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**batch)
            loss = outputs.loss
        loss.backward()
        grad_norm = self.model.clip_grad_norm_(self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler: self.scheduler.step()
        return {"loss": loss.item(), "grad_norm": grad_norm.item(),
                "lr": self.optimizer.param_groups[0]["lr"]}
