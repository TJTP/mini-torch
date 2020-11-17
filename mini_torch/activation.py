"""
激活函数 (激活层)
目前实现了Sigmoid, ReLU, Tanh和Linear
"""
from mini_torch.tensor import convert_to_tensor, exp, clip

class Activation():
    def __init__(self, name):
        self.name = name
        self.inputs = None
    
    def func(self, x):
        raise NotImplementedError

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__("Sigmoid")
    
    def func(self, x):
        return 1.0 / (1.0 + exp(-x, True))

class Tanh(Activation):
    def __init__(self):
        super().__init__("Tanh")

    def func(self, x):
        return (1.0 - exp(-x, True)) / 1.0 + exp(-x, True)
    
class ReLU(Activation):
    def __init__(self):
        super().__init__("ReLU")
    
    def func(self, x):
        return clip(x, low=0.0, requires_grad=True)

class Linear(Activation):
    def __init__(self):
        super().__init__("Linear")
    
    def func(self, x):
        return convert_to_tensor(x)