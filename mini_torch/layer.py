import numpy as np
import mini_torch.operation as ops 
from mini_torch.initializer import XavierInitializer, ConstInitializer
from mini_torch.tensor import convert_to_tensor

class Layer():
    def __init__(self, name):
        self.name = name 

        self.params, self.grads = {}, {}
        self._is_training = True
    
    def forward(self, inputs):
        raise NotImplementedError

    def set_status(self, is_training):
        self._is_training = is_training

class DenseLayer(Layer):
    def __init__(self, cur_layer_num, last_layer_num=None, 
                 w_initializer=XavierInitializer(), b_initializer=ConstInitializer()):
        
        super().__init__("Fully-connected-layer")
        self.initializer = {"w": w_initializer, "b": b_initializer}
        self.shapes = {"w": [last_layer_num, cur_layer_num], "b": [1, cur_layer_num]}
        self.params = {"w":None, "b":None}
        self._is_init = False
        if last_layer_num is not None:
            self._init_params(last_layer_num)

        self.inputs = None             
    
    def _init_params(self, last_layer_num):
        #print(last_layer_num)
        self.shapes["w"][0] = last_layer_num
        self.params["w"] = self.initializer["w"].init(self.shapes["w"])
        self.params["w"].zero_grad()
        self.params["b"] = self.initializer["b"].init(self.shapes["b"])
        self.params["b"].zero_grad()
        self._is_init = True
    
    def forward(self, inputs):
        if not self._is_init:
            self._init_params(inputs.shape[1]) # 等下修改
        
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

class Activation(Layer):
    def __init__(self, name):
        super().__init__(name)
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
        return 1.0 / (1.0 + ops.exp(-x, True))

class Tanh(Activation):
    def __init__(self):
        super().__init__("Tanh")

    def func(self, x):
        return (1.0 - ops.exp(-x, True)) / 1.0 + ops.exp(-x, True)
    
class ReLU(Activation):
    def __init__(self):
        super().__init__("ReLU")
    
    def func(self, x):
        return ops.clip(x, low=0.0, requires_grad=True)




