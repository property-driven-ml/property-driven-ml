import torch

from logic import Logic
    
class DL2(Logic):
    def __init__(self):
        super().__init__('DL2')

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x - y, min=0.)

    def LT(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xi = 1.
        return self.AND(self.LEQ(x, y), xi * (x == y).float())

    def NOT(self, _x: None) -> torch.Tensor:
        raise NotImplementedError('DL2 does not have general negation - rewrite the constraint to push negation inwards, e.g. NOT(x <= y) should be (y < x)')

    def AND(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

    def OR(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y