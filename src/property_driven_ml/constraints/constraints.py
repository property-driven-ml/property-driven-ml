import torch
import torch.nn.functional as F
import torch.linalg as LA

from abc import ABC, abstractmethod
from typing import Callable

from ..logics.logic import Logic
from ..logics.boolean_logic import BooleanLogic
from ..logics.fuzzy_logics import FuzzyLogic
from ..logics.stl import STL

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
    def eval(self, N: torch.nn.Module, x: torch.Tensor, x_adv: torch.Tensor, y_target: torch.Tensor | None, logic: Logic, reduction: str | None = None, skip_sat: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        constraint = self.get_constraint(N, x, x_adv, y_target)

        loss = constraint(logic)
        assert not torch.isnan(loss).any()

        if isinstance(logic, FuzzyLogic):
            loss = torch.ones_like(loss) - loss
        elif isinstance(logic, STL):
            loss = torch.clamp(logic.NOT(loss), min=0.)

        if skip_sat:
            # When skipping sat calculation, return a dummy tensor with same shape as loss
            sat = torch.zeros_like(loss)
        else:
            sat = constraint(self.boolean_logic).float()

        def agg(value: torch.Tensor) -> torch.Tensor:
            if reduction == None:
                return value
            elif reduction == 'mean':
                # Convert boolean tensors to float for mean calculation
                if value.dtype == torch.bool:
                    value = value.float()
                return torch.mean(value)
            elif reduction == 'sum':
                return torch.sum(value)
            else:
                raise ValueError(f'Unsupported reduction: {reduction}')

        return agg(loss), agg(sat)

class StandardRobustnessConstraint(Constraint):
    def __init__(self, device: torch.device, delta: float | torch.Tensor):
        super().__init__(device)

        assert 0. <= delta <= 1., 'delta is a probability and should be within the range [0, 1]'
        self.delta = torch.as_tensor(delta, device=self.device)

    def get_constraint(self, N: torch.nn.Module, x: torch.Tensor, x_adv: torch.Tensor, _y_target: None) -> Callable[[Logic], torch.Tensor]:
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
    def __init__(self, device: torch.device, lo: float | torch.Tensor | None, hi: float | torch.Tensor | None, normalize: bool = True):
        super().__init__(device)

        # Store raw bounds and normalization flag
        self.lo_raw = lo
        self.hi_raw = hi
        self.normalize = normalize
        
        # If normalization is disabled (for backwards compatibility), store as tensors directly
        if not normalize:
            self.lo = torch.as_tensor(lo, device=device) if lo is not None else None
            self.hi = torch.as_tensor(hi, device=device) if hi is not None else None

    def get_constraint(self, N: torch.nn.Module, _x: None, x_adv: torch.Tensor, _y_target: None) -> Callable[[Logic], torch.Tensor]:
        y_adv = N(x_adv).squeeze()

        if self.normalize:
            # Import here to avoid circular imports
            from examples.alsomitra_dataset import AlsomitraDataset
            
            # Normalize bounds at constraint time using class constants
            lo_normalized = None if self.lo_raw is None else (torch.tensor(self.lo_raw, device=self.device) - AlsomitraDataset.C_out) / AlsomitraDataset.S_out
            hi_normalized = None if self.hi_raw is None else (torch.tensor(self.hi_raw, device=self.device) - AlsomitraDataset.C_out) / AlsomitraDataset.S_out
            
            lo_normalized = lo_normalized.squeeze() if lo_normalized is not None else None
            hi_normalized = hi_normalized.squeeze() if hi_normalized is not None else None
        else:
            # Use pre-normalized bounds
            lo_normalized = self.lo
            hi_normalized = self.hi

        if lo_normalized is None and hi_normalized is not None:
            return lambda l: l.LEQ(y_adv, hi_normalized)
        elif lo_normalized is not None and hi_normalized is not None:
            return lambda l: l.AND(l.LEQ(lo_normalized, y_adv), l.LEQ(y_adv, hi_normalized))
        elif lo_normalized is not None and hi_normalized is None:
            return lambda l: l.LEQ(lo_normalized, y_adv)
        else:
            raise ValueError('need to specify either lower or upper (or both) bounds for e_x') 

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