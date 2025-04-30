from __future__ import print_function

import torch

# GradNorm (https://arxiv.org/abs/1711.02257) based on https://github.com/brianlan/pytorch-grad-norm/blob/master/train.py
class GradNorm():
    def __init__(self, N: torch.nn.Module, device: torch.device, optimizer, lr: float, alpha: float, initial_dl_weight=1.):
        self.initial_loss = None
        self.weights = torch.tensor([2. - initial_dl_weight, initial_dl_weight], requires_grad=True)
        self.N = N
        self.device = device
        self.optimizer_train = optimizer
        self.optimizer_weights = torch.optim.Adam([self.weights], lr=lr)
        self.alpha = alpha

    def balance(self, ce_loss: torch.Tensor, dl_loss: torch.Tensor):
        task_loss = torch.stack([ce_loss, dl_loss])

        if self.initial_loss is None:
            initial_loss = task_loss.detach()

            # prevent division by zero later
            self.initial_loss = torch.where(initial_loss == 0., torch.finfo(initial_loss.dtype).eps, initial_loss)

        weighted_task_loss = self.weights[0] * ce_loss + self.weights[1] * dl_loss  
        weighted_task_loss.backward()

        self.optimizer_train.step()

        norms = []

        for weight in self.weights:
            norms.append(weight * torch.sqrt(sum(p.grad.norm() ** 2 for p in self.N.parameters() if p.grad is not None)))

        norms = torch.stack(norms)

        loss_ratio = torch.stack([ce_loss.detach(), dl_loss.detach()]) / self.initial_loss
        inverse_train_rate = loss_ratio / loss_ratio.mean()

        mean_norm = norms.mean()
        constant_term = mean_norm * (inverse_train_rate ** self.alpha)
        grad_norm_loss = (norms - constant_term).abs().sum()

        self.optimizer_weights.zero_grad(set_to_none=True)
        grad_norm_loss.backward()
        self.optimizer_weights.step()

    @torch.no_grad
    def renormalise(self):
        normalise_coeff = 2. / torch.sum(self.weights.data, dim=0)
        self.weights.data = self.weights.data * normalise_coeff

        print(f'GradNorm weights={self.weights.data}')