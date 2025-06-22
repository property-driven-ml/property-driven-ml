import torch

from abc import ABC, abstractmethod
from functools import reduce

class Logic(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def AND(self, *xs: torch.Tensor) -> torch.Tensor:
        if len(xs) < 2:
            raise ValueError('AND requires at least 2 arguments. If you have a list xs, make sure to unpack it with *xs.')
        
        return reduce(self.AND2, xs)

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:        
        raise NotImplementedError('AND2 must be implemented if AND is not overridden.')

    def OR(self, *xs: torch.Tensor) -> torch.Tensor:
        if len(xs) < 2:
            raise ValueError('OR requires at least 2 arguments. If you have a list xs, make sure to unpack it with *xs.')
        
        return reduce(self.OR2, xs)

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('OR2 must be implemented if OR is not overridden.')

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.OR(self.NOT(x), y)

    def EQUIV(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.AND(self.IMPL(x, y), self.IMPL(y, x))