import math
from torch.optim import Optimizer

class CosineScheduler:
    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, min_lr: float = 5e-6, max_lr: float = 3e-4):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        if self.current_step < self.num_warmup_steps:
            lr = self.max_lr * (self.current_step / self.num_warmup_steps)
        else:
            progress = (self.current_step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1