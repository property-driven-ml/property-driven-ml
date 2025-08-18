from __future__ import print_function

from abc import ABC, abstractmethod

import torch

from ..logics.logic import Logic
from ..constraints.constraints import Constraint

class Attack(ABC):
    def __init__(self, x0: torch.Tensor, logic: Logic, device: torch.device, steps: int, restarts: int, mean: torch.Tensor | tuple[float, ...] = (0.,), std: torch.Tensor | tuple[float, ...] = (1.,)):
        self.logic = logic
        self.device = device
        self.steps = steps
        self.restarts = restarts
        self.mean = torch.as_tensor(mean, device=device)
        self.std = torch.as_tensor(std, device=device)

        self.ndim = x0.ndim
        expand = lambda t: t.view(*t.shape, *([1] * (self.ndim - t.ndim)))

        self.min = expand((0. - self.mean) / self.std)
        self.max = expand((1. - self.mean) / self.std)
    
    def uniform_random_sample(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        z = torch.empty_like(lo).uniform_(0., 1.) * (hi - lo) + lo
        return torch.clamp(z, min=self.min, max=self.max)

    @abstractmethod
    def attack(self, N: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, bounds: tuple[torch.Tensor, torch.Tensor], constraint: Constraint) -> torch.Tensor:
        pass

class PGD(Attack):
    def __init__(self, x0: torch.Tensor, logic: Logic, device: torch.device, steps: int, restarts: int, step_size: float, mean: torch.Tensor | tuple[float, ...] = (0.,), std: torch.Tensor | tuple[float, ...] = (1.,)):
        super().__init__(x0, logic, device, steps, restarts, mean, std)
        self.step_size = step_size / torch.as_tensor(std, device=device)

        print(f'PGD steps={self.steps} restarts={self.restarts} step_size={self.step_size}')

    @torch.enable_grad
    def attack_single(self, N: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, constraint: Constraint, random_start: bool = True) -> torch.Tensor:
        x = x.clone().detach()
        x_adv = x.clone().detach()

        if random_start:
            x_adv = self.uniform_random_sample(lo, hi).detach()

        for _ in range(self.steps):
            x_adv.requires_grad_(True)

            loss, _ = constraint.eval(N, x, x_adv, y, self.logic, reduction='mean', skip_sat=True)

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                x_adv.add_(self.step_size * grad.sign_())
                x_adv = torch.max(torch.min(x_adv, hi), lo)
                x_adv.clamp_(min=self.min, max=self.max)

            x_adv = x_adv.detach()

        return x_adv, loss

    def attack(self, N: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, constraint: Constraint) -> torch.Tensor:
        best_adv = None
        best_loss = None

        before = N.training
        N.train(True)

        for _ in range(self.restarts + 1):
            x_adv, loss = self.attack_single(N, x, y, lo, hi, constraint)

            if best_loss is None or best_loss < loss:
                best_adv = x_adv
                best_loss = loss

        N.train(before)

        return best_adv
    
# AutoPGD (https://arxiv.org/abs/2003.01690) based on https://github.com/fra31/auto-attack/blob/master/autoattack/autopgd_base.py
class APGD(Attack):
    def __init__(self, x0: torch.Tensor, logic: Logic, device: torch.device, steps: int, restarts: int, mean: torch.Tensor | tuple[float, ...] = (0.,), std: torch.Tensor | tuple[float, ...] = (1.,), rho: float = .75):
        super().__init__(x0, logic, device, steps, restarts, mean, std)
        self.rho = rho

        self.eot_iter = 1

        self.n_iter2 = max(int(.22 * self.steps), 1)
        self.n_iter_min = max(int(.06 * self.steps), 1)
        self.size_decr = max(int(.03 * self.steps), 1)

    def check_oscillation(self, x, j, k, k3 = .75):
        t = torch.zeros(x.shape[1]).to(self.device)

        for counter5 in range(k):
            t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def attack_single(self, N: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, constraint: Constraint) -> tuple[torch.Tensor, torch.Tensor]:
        x_adv = self.uniform_random_sample(lo, hi).detach()
        x_best = x_adv.clone()

        loss_steps = torch.zeros([self.steps, x.shape[0]]).to(self.device)

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)

        for _ in range(self.eot_iter):
            with torch.enable_grad():
                loss_indiv, _ = constraint.eval(N, x, x_adv, y, self.logic, reduction=None, skip_sat=True)
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        loss_best = loss_indiv.detach().clone()

        step_size = (hi - lo) * torch.ones([x.shape[0], *([1] * self.ndim)]).to(self.device).detach()
        x_adv_old = x_adv.clone()

        k = self.n_iter2 + 0

        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        for i in range(self.steps):
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = .75 if i > 0 else 1.

                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, lo), hi), self.min, self.max)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1. - a), lo), hi), self.min, self.max)

                x_adv = x_adv_1 + 0.

            x_adv.requires_grad_()
            grad = torch.zeros_like(x)

            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    loss_indiv, _ = constraint.eval(N, x, x_adv, y, self.logic, reduction=None, skip_sat=True)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(loss_steps, i, k, k3=self.rho)
                    fl_reduce_no_impr = (1. - reduced_last_check) * (loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.

                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    k = max(k - self.size_decr, self.n_iter_min)

        return x_best, loss_best

    def attack(self, N: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, constraint: Constraint) -> torch.Tensor:
        before = N.training
        N.train(True)

        x = x.detach().clone().float().to(self.device)
        y = y.detach().clone().long().to(self.device)

        adv_best = x.detach().clone()
        loss_best = torch.ones([x.shape[0]]).to(self.device) * (-torch.inf)

        for _ in range(self.restarts + 1):
            best_curr, loss_curr = self.attack_single(N, x, y, lo, hi, constraint)

            i = (loss_curr > loss_best).nonzero().squeeze()
            adv_best[i] = best_curr[i] + 0.
            loss_best[i] = loss_curr[i] + 0.

        N.train(before)

        return adv_best