import numpy as np 

class Tensor():
    def __init__(self, values, requires_grad=False, dependencies=[]):
        self._values = np.array(values)
        
        self.grad = None
        self.requires_grad = requires_grad
        if self.requires_grad:
            self.zero_grad()
        
        self.dependencies = dependencies
    
    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, new_values):
        self._values = np.array(new_values)
        self.grad = None
    
    @property
    def shape(self):
        return self._values.shape

    @property
    def T(self):
        return

    def zero_grad(self):
        self.grad = np.zeros(self.shape)
    
    def backward(self, grad=None):
        assert self.requires_grad, "Run backward() on a non-requires-grad tensor!"

        gradient = 1.0 if grad == None else grad
        self.grad += np.array(gradient) #梯度累积

        for dependency in self.dependencies:
            grad_for_dep = dependency["grad_fn"](gradient)
            dependency["tensor"].backward(grad_for_dep)
    