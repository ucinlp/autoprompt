import logging

import torch


logger = logging.getLogger(__name__)


class L1SGD(torch.optim.Optimizer):
    """Truncated Stochastic Gradient Descent w/ L1 Penalty.

    For details see: https://arxiv.org/abs/0806.4686
    """
    def __init__(self, params, lr, l1decay=0.0, theta=1e32):
        default = dict(lr=lr, l1decay=l1decay, theta=theta)
        super().__init__(params, default)

    @staticmethod
    def _update(params, d_p_list, lr, l1decay, theta):
        for i, param in enumerate(params):
            # Standard SGD Step
            d_p = d_p_list[i]
            zeros = param.eq(0.0)
            param.add_(d_p, alpha=-lr)

            # Truncation
            if l1decay != 0.0:
                logger.debug('Truncated routine invoked.')
                branch1 = param.ge(0.0) & param.le(theta)
                branch2 = param.le(0.0) & param.ge(-theta)
                # Max 0
                branch1_update = param[branch1] - lr*l1decay
                branch1_mask = branch1_update.lt(0.0)
                branch1_update[branch1_mask] = 0.0
                param[branch1] = branch1_update
                # Min 0
                branch2_update = param[branch2] + lr*l1decay
                branch2_mask = branch2_update.gt(0.0)
                branch2_update[branch2_mask] = 0.0
                param[branch2] = branch2_update

                # Make sure zeros stay zeros
                param[zeros] = 0.0

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']
            l1decay = group['l1decay']
            theta = group['theta']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            self._update(
                params_with_grad,
                d_p_list,
                lr,
                l1decay,
                theta,
            )
