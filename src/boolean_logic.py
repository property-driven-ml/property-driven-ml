import torch

from logic import Logic

class BooleanLogic(Logic):
    def __init__(self):
        super().__init__('bool')

    def LEQ(self, x, y):
        return x <= y

    def NOT(self, x):
        return torch.logical_not(x)

    def AND(self, x, y):
        return torch.logical_and(x, y)

    def OR(self, x, y):
        return torch.logical_or(x, y)

    def IMPL(self, x, y):
        return torch.logical_or(torch.logical_not(x), y)