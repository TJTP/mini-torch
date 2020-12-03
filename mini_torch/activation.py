"""
激活函数 (激活层)
目前实现了Sigmoid, ReLU, Tanh和Linear
"""
from mini_torch.tensor import convert_to_tensor, exp, clip

class BaseActivation:
    def __init__(self, name, show=False):
        self.params = {}#定义空字典, 因为在应用中将激活函数也看作
                        #一层, 只不过没有参数
        self._inputs = None
        self.show = show
        self.name = name + "-act"
    
    def func(self, x):
        raise NotImplementedError

    def forward(self, inputs):
        self._inputs = inputs
        output = self.func(self._inputs)
        if self.show:
            print('<'+self.name+'>')
            print(output.values)
        return output

class Sigmoid(BaseActivation):
    def __init__(self, show=False):
        super().__init__("Sigmoid", show=show)
    
    def func(self, x):
        return 1.0 / (1.0 + exp(-x, True))

class Tanh(BaseActivation):
    def __init__(self, show=False):
        super().__init__("Tanh", show=show)
    
    def func(self, x):
        return (1.0 - exp(-x, True)) / 1.0 + exp(-x, True)
    
class ReLU(BaseActivation):
    def __init__(self, show=False):
        super().__init__("ReLU", show=show)
    
    def func(self, x):
        return clip(x, low=0.0, requires_grad=True)

class Linear(BaseActivation):
    def __init__(self, show=False):
        super().__init__("Linear", show=show)
    
    def func(self, x):
        return convert_to_tensor(x)