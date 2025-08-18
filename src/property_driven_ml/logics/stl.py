import torch

from .logic import Logic

from ..util import safe_div
    
class STL(Logic):
    def __init__(self, k=1.):
        super().__init__('STL')
        self.k = k

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y - x

    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        return -x
    
    def AND(self, *xs) -> torch.Tensor:
        xs = torch.stack(xs)
        x_min, _ = torch.min(xs, dim=0)
        rel = safe_div(xs - x_min, x_min)

        # NOTE: for numerical stability do not directly calculate exp

        # case 1: x_min < 0
        rel_max = rel.max(dim=0, keepdim=True).values
        exp1 = torch.exp(rel - rel_max)
        exp2 = torch.exp(self.k * rel - self.k * rel_max)

        num = (x_min * exp1 * exp2).sum(dim=0)
        denom = exp1.sum(dim=0)
        neg = safe_div(num, denom)

        # case 2: x_min > 0
        krel = -self.k * rel
        krel_max = krel.max(dim=0, keepdim=True).values
        exp_krel = torch.exp(krel - krel_max)

        num = (xs * exp_krel).sum(dim=0)
        denom = exp_krel.sum(dim=0)
        pos = safe_div(num, denom)

        return torch.where(x_min < 0, neg, torch.where(x_min > 0, pos, x_min))

    def OR(self, *xs) -> torch.Tensor:
        return self.NOT(self.AND(*[self.NOT(x) for x in xs]))