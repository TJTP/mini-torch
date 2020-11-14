import numpy as np 

class BaseOptimizer():
    def __init__(self, learning_rate, weight_decay):
        self.lr = learning_rate
        self.weight_decay = weight_decay
    
    def _compute_step(self, grad):
        return NotImplementedError

class SGD(BaseOptimizer):
    def __init__(self, learning_rate, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
    
    def _compute_step(self, grad):
        return -self.lr * grad

