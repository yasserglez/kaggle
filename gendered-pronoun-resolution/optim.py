import math
import logging
import numpy as np

import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_


logger = logging.getLogger(__name__)


def lr_schedule(step, total_steps, lr, lr_warmup, lr_cooldown):
    """Calculate a scheduled learning rate value."""
    if lr_warmup is None:
        lr_warmup = 0
    else:
        if 0 < lr_warmup <= 1:
            lr_warmup = math.ceil(lr_warmup * total_steps)
        if step <= lr_warmup:
            return lr * (step / lr_warmup)
    if lr_cooldown == 'linear':
        return lr * (1.0 - (step - lr_warmup) /
                           (total_steps - lr_warmup))
    else:
        return lr


class Adam(Optimizer):
    """
    Implementation of the Adam optimization algorithm including ideas from:

    - Adam: A Method for Stochastic Optimization (https://arxiv.org/abs/1412.6980)
    - Decoupled Weight Decay Regularization (https://arxiv.org/abs/1711.05101)
    - https://github.com/huggingface/pytorch-pretrained-BERT/
    """

    def __init__(self, named_parameters, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=None, max_grad_norm=None,
                 lr_warmup=None, lr_cooldown=None, total_steps=None):
        """
        Arguments:
            named_parameters: Iterable yielding both the name of the
                parameter as well as the parameter itself.
            lr: Initial learning rate.
            beta1, beta2: Exponential decay rates for moving average of
                gradient values (beta1) and square gradient values (beta2).
            eps: Term added to the denominator of the update rule
                to improve numerical stability.
            weight_decay: Weight decay factor. None means no decay.
            max_grad_norm: Maximum norm for gradient clipping. None means no clipping.
            lr_warmup: Linearly increase the learning rate for the first steps.
                Supported values: None (disabled), int (number of warmup steps),
                float in (0,1] (warmup steps as a ratio of total_steps).
            lr_cooldown: Schedule followed to reduce the learning rate.
                Supported values: None (disabled), 'linear' (decrease it linearly
                to zero after total_steps steps are completed).
            total_steps: Total number of parameter update steps.
                Required for certain lr_cooldown schedules.
        """
        if not lr >= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        for beta in (beta1, beta2):
            if not 0.0 <= beta <= 1.0:
                raise ValueError('Invalid beta value: {}'.format(beta))
        if not eps >= 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not (weight_decay is None or 0.0 <= weight_decay <= 1.0):
            raise ValueError('Invalid weight decay: {}'.format(weight_decay))
        if not (max_grad_norm is None or max_grad_norm >= 0.0):
            raise ValueError('Invalid maximum norm for gradient clipping: {}'.format(max_grad_norm))
        if not (total_steps is None or total_steps > 0):
            raise ValueError('Invalid total number of steps: {}'.format(total_steps))

        if not (lr_warmup is None or lr_warmup >= 0.0):
            raise ValueError('Invalid learning rate warmup: {}'.format(lr_warmup))
        if lr_warmup is not None and 0 < lr_warmup <= 1 and total_steps is None:
            raise ValueError('total_steps is required if 0 < lr_warmup <= 1')

        if lr_cooldown not in (None, 'linear'):
            raise ValueError('Invalid learning rate cooldown: {}'.format(lr_cooldown))
        if lr_cooldown == 'linear' and total_steps is None:
            raise ValueError("total_steps is required if lr_cooldown is 'linear'")

        # Collect the parameters
        param_count = 0
        with_weight_decay, without_weight_decay = [], []
        for name, param in named_parameters:
            if param.requires_grad:
                param_size = np.prod(param.size())
                param_count += param_size
                if weight_decay is not None and \
                        name.endswith('.weight') and 'norm' not in name:
                    with_weight_decay.append(param)
                    logger.info('Parameter: %s (size = %d, weight decay = %g)',
                                name, param_size, weight_decay)
                else:
                    without_weight_decay.append(param)
                    logger.info('Parameter: %s (size = %d, weight decay = None)',
                                name, param_size)
        param_groups = [
            {'params': with_weight_decay, 'weight_decay': weight_decay},
            {'params': without_weight_decay, 'weight_decay': None},
        ]
        logger.info('Optimizing %d parameters', param_count)

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, max_grad_norm=max_grad_norm,
                        lr_warmup=lr_warmup, lr_cooldown=lr_cooldown,
                        total_steps=total_steps)
        super().__init__(param_groups, defaults)

    def step(self, closure=None):
        """Perform a single parameter update step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values (m)
                    state['grad_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values (v)
                    state['squared_grad_avg'] = torch.zeros_like(p.data)

                beta1, beta2 = group['beta1'], group['beta2']
                grad_avg, squared_grad_avg = state['grad_avg'], state['squared_grad_avg']

                # Gradient clipping
                if group['max_grad_norm'] is not None:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficients
                grad = p.grad.data
                grad_avg.mul_(beta1).add_(1 - beta1, grad)
                squared_grad_avg.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Calculate the effective step size
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                scheduled_lr = lr_schedule(
                    state['step'], group['total_steps'],
                    group['lr'], group['lr_warmup'], group['lr_cooldown'])
                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                # Update the parameters
                denom = squared_grad_avg.sqrt().add_(group['eps'])
                p.data.addcdiv_(-step_size, grad_avg, denom)
                if group['weight_decay'] is not None:
                    p.data.add_(-group['weight_decay'], p.data)

        return loss
