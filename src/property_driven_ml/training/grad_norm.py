from __future__ import print_function

import torch


# GradNorm (https://arxiv.org/abs/1711.02257) based on https://github.com/brianlan/pytorch-grad-norm/blob/master/train.py
class GradNorm:
    """Gradient normalization for multi-task learning.

    Implements the GradNorm algorithm to automatically balance multiple
    loss terms during training by adjusting their relative weights.

    Args:
        N: Neural network model.
        device: PyTorch device for computations.
        optimizer: Optimizer for model parameters.
        lr: Learning rate for weight optimization.
        alpha: Restoring force strength for loss balancing.
        initial_dl_weight: Initial weight for the second loss term.
    """

    def __init__(
        self,
        N: torch.nn.Module,
        device: torch.device,
        optimizer,
        lr: float,
        alpha: float,
        initial_dl_weight=1.0,
    ):
        self.initial_loss = None
        self.weights = torch.tensor(
            [2.0 - initial_dl_weight, initial_dl_weight], requires_grad=True
        )
        self.N = N
        self.device = device
        self.optimizer_train = optimizer
        self.optimizer_weights = torch.optim.Adam([self.weights], lr=lr)
        self.alpha = alpha

    def balance(self, ce_loss: torch.Tensor, dl_loss: torch.Tensor):
        """Balance the weights between cross-entropy and DL losses.

        Uses gradient magnitudes to automatically adjust loss weights
        for more stable multi-task training.

        Args:
            ce_loss: Cross-entropy loss tensor.
            dl_loss: Deep learning constraint loss tensor.

        Returns:
            Tuple of (weighted_ce_loss, weighted_dl_loss).
        """
        task_loss = torch.stack([ce_loss, dl_loss])

        if self.initial_loss is None:
            initial_loss = task_loss.detach()

            # prevent division by zero later
            self.initial_loss = torch.where(
                initial_loss == 0.0, torch.finfo(initial_loss.dtype).eps, initial_loss
            )

        weighted_task_loss = self.weights[0] * ce_loss + self.weights[1] * dl_loss
        weighted_task_loss.backward()

        self.optimizer_train.step()

        norms = []

        for weight in self.weights:
            # Collect gradient norms for all parameters with gradients
            grad_norms = torch.stack(
                [p.grad.norm() ** 2 for p in self.N.parameters() if p.grad is not None]
            )

            # Use torch.sum instead of built-in sum to ensure tensor output
            total_grad_norm_squared = (
                torch.sum(grad_norms)
                if len(grad_norms) > 0
                else torch.tensor(0.0, device=self.device)
            )

            norms.append(weight * torch.sqrt(total_grad_norm_squared))

        norms = torch.stack(norms)

        loss_ratio = (
            torch.stack([ce_loss.detach(), dl_loss.detach()]) / self.initial_loss
        )
        inverse_train_rate = loss_ratio / loss_ratio.mean()

        mean_norm = norms.mean()
        constant_term = mean_norm * (inverse_train_rate**self.alpha)
        grad_norm_loss = (norms - constant_term).abs().sum()

        self.optimizer_weights.zero_grad(set_to_none=True)
        grad_norm_loss.backward()
        self.optimizer_weights.step()

    @torch.no_grad
    def renormalise(self):
        normalise_coeff = 2.0 / torch.sum(self.weights.data, dim=0)
        self.weights.data = self.weights.data * normalise_coeff

        print(f"GradNorm weights={self.weights.data}")
