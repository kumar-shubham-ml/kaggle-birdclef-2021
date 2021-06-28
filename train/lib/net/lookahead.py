# 'Lookahead Optimizer: k steps forward, 1 step back' - Michael Zhang

# https://arxiv.org/pdf/1907.08610.pdf
# https://www.youtube.com/watch?v=ypqf7UUird4
# https://github.com/michaelrzhang/lookahead/blob/master/lookahead_pytorch.py

from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer
import itertools as it


# class Lookahead(Optimizer):
#     r"""PyTorch implementation of the lookahead wrapper.
#
#     Lookahead Optimizer: https://arxiv.org/abs/1907.08610
#     """
#
#     def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
#         """optimizer: inner optimizer
#         la_steps (int): number of lookahead steps
#         la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
#         pullback_momentum (str): change to inner optimizer momentum on interpolation update
#         """
#         self.optimizer = optimizer
#         self._la_step = 0  # counter for inner optimizer
#         self.la_alpha = la_alpha
#         self._total_la_steps = la_steps
#         pullback_momentum = pullback_momentum.lower()
#         assert pullback_momentum in ["reset", "pullback", "none"]
#         self.pullback_momentum = pullback_momentum
#
#         self.state = defaultdict(dict)
#
#         # Cache the current optimizer parameters
#         for group in optimizer.param_groups:
#             for p in group['params']:
#                 param_state = self.state[p]
#                 param_state['cached_params'] = torch.zeros_like(p.data)
#                 param_state['cached_params'].copy_(p.data)
#                 if self.pullback_momentum == "pullback":
#                     param_state['cached_mom'] = torch.zeros_like(p.data)
#
#     def __getstate__(self):
#         return {
#             'state': self.state,
#             'optimizer': self.optimizer,
#             'la_alpha': self.la_alpha,
#             '_la_step': self._la_step,
#             '_total_la_steps': self._total_la_steps,
#             'pullback_momentum': self.pullback_momentum
#         }
#
#     def zero_grad(self):
#         self.optimizer.zero_grad()
#
#     def get_la_step(self):
#         return self._la_step
#
#     def state_dict(self):
#         return self.optimizer.state_dict()
#
#     def load_state_dict(self, state_dict):
#         self.optimizer.load_state_dict(state_dict)
#
#     def _backup_and_load_cache(self):
#         """Useful for performing evaluation on the slow weights (which typically generalize better)
#         """
#         for group in self.optimizer.param_groups:
#             for p in group['params']:
#                 param_state = self.state[p]
#                 param_state['backup_params'] = torch.zeros_like(p.data)
#                 param_state['backup_params'].copy_(p.data)
#                 p.data.copy_(param_state['cached_params'])
#
#     def _clear_and_load_backup(self):
#         for group in self.optimizer.param_groups:
#             for p in group['params']:
#                 param_state = self.state[p]
#                 p.data.copy_(param_state['backup_params'])
#                 del param_state['backup_params']
#
#     @property
#     def param_groups(self):
#         return self.optimizer.param_groups
#
#     def step(self, closure=None):
#         """Performs a single Lookahead optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = self.optimizer.step(closure)
#         self._la_step += 1
#
#         if self._la_step >= self._total_la_steps:
#             self._la_step = 0
#             # Lookahead and cache the current optimizer parameters
#             for group in self.optimizer.param_groups:
#                 for p in group['params']:
#                     param_state = self.state[p]
#                     p.data.mul_(self.la_alpha).add_(1.0 - self.la_alpha, param_state['cached_params'])  # crucial line
#                     param_state['cached_params'].copy_(p.data)
#                     if self.pullback_momentum == "pullback":
#                         internal_momentum = self.optimizer.state[p]["momentum_buffer"]
#                         self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
#                             1.0 - self.la_alpha, param_state["cached_mom"])
#                         param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
#                     elif self.pullback_momentum == "reset":
#                         self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)
#
#         return loss
#

# Lookahead implementation from https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py

class Lookahead(Optimizer):
    def __init__(self, optimizer, alpha=0.5, k=6):

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')

        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0

        self.slow_weights = [
                [p.clone().detach() for p in group['params']]
            for group in self.param_groups]

        for w in it.chain(*self.slow_weights):
            w.requires_grad = False
        self.state = optimizer.state

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()

        for group,slow_weights in zip(self.param_groups,self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p,q in zip(group['params'],slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(p.data - q.data, alpha=self.alpha )
                p.data.copy_(q.data)
        return loss