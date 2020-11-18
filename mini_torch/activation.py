"""
激活函数 (激活层)
目前实现了Sigmoid, ReLU, Tanh和Linear
"""
from mini_torch.tensor import convert_to_tensor, exp, clip

class BaseActivation:
    def __init__(self):
        self.params = {}#定义空字典, 因为在应用中将激活函数也看作
                        #一层, 只不过没有参数
        self.inputs = None
    
    def func(self, x):
        raise NotImplementedError

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

class Sigmoid(BaseActivation):
    
    def func(self, x):
        return 1.0 / (1.0 + exp(-x, True))

class Tanh(BaseActivation):
    
    def func(self, x):
        return (1.0 - exp(-x, True)) / 1.0 + exp(-x, True)
    
class ReLU(BaseActivation):
    
    def func(self, x):
        return clip(x, low=0.0, requires_grad=True)

class Linear(BaseActivation):
    
    def func(self, x):
        return convert_to_tensor(x)