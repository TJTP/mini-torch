import numpy as np
import mini_torch.operation as ops 
from mini_torch.initializer import XavierInitializer, ConstInitializer
from mini_torch.tensor import convert_to_tensor

class Layer():
    def __init__(self, name):
        self.name = name 

        self.params, self.grads = {}, {}
        self.is_training = True
    
    def foward(self, inputs):
        raise NotImplementedError

    def set_status(self, is_training):
        self.is_training = True if is_training == True else False

class DenseLayer(Layer):
    def __init__(self, cul_layer_num, last_layer_num=None, 
                 w_initializer=XavierInitializer(), b_initializer=ConstInitializer()):
        
        super().__init__("Fully-connected-layer")
        self.initializer = {"w": w_initializer, "b": b_initializer}
        self.shapes = {"w": [last_layer_num, cul_layer_num], "b": [1, last_layer_num]}
        self.params = {"w":None, "b":None}
        self._is_init = False
        if last_layer_num is not None:
            self._init_params(last_layer_num)

        self.inputs = None             
    
    def _init_params(self, last_layer_num):
        self.shapes["w"][0] = last_layer_num
        self.params["w"] = self.initializer["w"].init(self.shapes["w"])
        self.params["w"].zero_grad()
        self.params["b"] = self.initializer["b"].init(self.shapes["b"])
        self.params["b"].zero_grad()
        self._is_init = True
    
    def foward(self, inputs):
        if not self._is_init:
            self._init_params(inputs) # 等下修改
        
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

class Activation(Layer):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = None
    
    def func(self, x):
        raise NotImplementedError

    def foward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__("Sigmoid")
    
    def func(self, x):
        tensor_x = convert_to_tensor(x)
        return 1.0 / (1.0 + ops.exp_(-tensor_x))

class Tanh(Activation):
    def __init__(self):
        super().__init__("Tanh")

    def func(self, x):
        tensor_x = convert_to_tensor(x)
        return (1.0 - ops.exp_(-tensor_x)) / 1.0 + ops.exp_(-tensor_x)
    
class ReLU(Activation):
    def __init__(self):
        super().__init__("ReLU")
    
    def func(self, x):
        tensor_x = convert_to_tensor(x)
        return ops.clip_(tensor_x, low=0.0)




