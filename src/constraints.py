import torch
import torch.nn.functional as F
import torch.linalg as LA

from abc import ABC, abstractmethod
from typing import Callable

from logic import Logic
from boolean_logic import BooleanLogic
from fuzzy_logics import FuzzyLogic
from stl import STL

class Constraint(ABC):
    def __init__(self, device: torch.device):
        self.device = device
        self.boolean_logic = BooleanLogic()

    @abstractmethod
    def get_constraint(self, N: torch.nn.Module, x: torch.Tensor | None, x_adv: torch.Tensor | None, y_target: torch.Tensor | None) -> Callable[[Logic], torch.Tensor]:
        pass

    # usage:
    # loss, sat = eval()
    # sat indicates whether the constraint is satisfied or not
    def eval(self, N: torch.nn.Module, x: torch.Tensor, x_adv: torch.Tensor, y_target: torch.Tensor, logic: Logic, reduction: str | None = None, skip_sat: bool = False) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        constraint = self.get_constraint(N, x, x_adv, y_target)
        loss, sat = None, None

        loss = constraint(logic)
        assert not torch.isnan(loss).any()

        if isinstance(logic, FuzzyLogic):
            loss = torch.ones_like(loss) - loss
        elif isinstance(logic, STL):
            loss = torch.clamp(logic.NOT(loss), min=0.)

        if not skip_sat:
            sat = constraint(self.boolean_logic).float()

        def agg(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None

            if reduction == None:
                return value
            elif reduction == 'mean':
                return torch.mean(value)
            elif reduction == 'sum':
                return torch.sum(value)
            else:
                assert False, f'unsupported reduction {reduction}'

        return agg(loss), agg(sat)

class StandardRobustnessConstraint(Constraint):
    def __init__(self, device: torch.device, delta: float | torch.Tensor):
        super().__init__(device)

        assert 0. <= delta <= 1., 'delta is a probability and should be within the range [0, 1]'
        self.delta = torch.as_tensor(delta, device=self.device)

    def get_constraint(self, N: torch.nn.Module, x: torch.Tensor, x_adv: torch.Tensor, _y_target: None) -> Callable[[Logic], torch.tensor]:
        y = N(x)
        y_adv = N(x_adv)

        diff = F.softmax(y_adv, dim=1) - F.softmax(y, dim=1)

        return lambda l: l.LEQ(LA.vector_norm(diff, ord=float('inf'), dim=1), self.delta)
    
class LipschitzRobustnessConstraint(Constraint):
    def __init__(self, device: torch.device, L: float):
        super().__init__(device)

        self.L = torch.as_tensor(L, device=device)

    def get_constraint(self, N: torch.nn.Module, x: torch.Tensor, x_adv: torch.Tensor, _y_target: None) -> Callable[[Logic], torch.Tensor]:
        y = N(x)
        y_adv = N(x_adv)

        diff_x = LA.vector_norm(x_adv - x, ord=2, dim=1)
        diff_y = LA.vector_norm(y_adv - y, ord=2, dim=1)

        return lambda l: l.LEQ(diff_y, self.L * diff_x)
    
class AlsomitraOutputConstraint(Constraint):
    def __init__(self, device: torch.device, lo: float | torch.Tensor, hi: float | torch.Tensor):
        super().__init__(device)

        self.lo = torch.as_tensor(lo, device=device)
        self.hi = torch.as_tensor(hi, device=device)

    def get_constraint(self, N: torch.nn.Module, _x: None, x_adv: torch.Tensor, _y_target: None) -> Callable[[Logic], torch.Tensor]:
        y_adv = N(x_adv).squeeze()

        if self.lo is None and self.hi is not None:
            return lambda l: l.LEQ(y_adv, self.hi)
        elif not self.lo is None and self.hi is not None:
            return lambda l: l.AND(l.LEQ(self.lo, y_adv), l.LEQ(y_adv, self.hi))
        elif self.lo is not None and self.hi is None:
            return lambda l: l.LEQ(self.lo, y_adv)
        else:
            assert False, 'need to specify either lower or upper (or both) bounds for e_x'

class GroupConstraint(Constraint):
    def __init__(self, device: torch.device, indices: list[list[int]], delta: float):
        super().__init__(device)

        self.indices = indices

        assert 0. <= delta <= 1., 'delta is a probability and should be within the range [0, 1]'
        self.delta = torch.as_tensor(delta, device=self.device)

    def get_constraint(self, N: torch.nn.Module, _x: None, x_adv: torch.Tensor, _y_target: None) -> Callable[[Logic], torch.Tensor]:
        y_adv = F.softmax(N(x_adv), dim=1)
        sums = [torch.sum(y_adv[:, i], dim=1) for i in self.indices]

        return lambda l: l.AND(*[l.OR(l.LEQ(s, self.delta), l.LEQ(1. - self.delta, s)) for s in sums])