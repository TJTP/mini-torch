import numpy as np 

class BaseOptimizer():
    def __init__(self, learning_rate, weight_decay):
        self.lr = learning_rate
        self.weight_decay = weight_decay
    
    def compute_step(self, grads_each_layer):
        steps_each_layer = []
        for grads in grads_each_layer:
            steps_each_layer.append(self._compute_step(grads))

        return steps_each_layer
    
    def _compute_step(self, grad):
        return NotImplementedError

class SGD(BaseOptimizer):
    def __init__(self, learning_rate, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
    
    def _compute_step(self, grads):
        for key, value in grads.items():
            grads[key] = -grads[key] * self.lr
        return grads

