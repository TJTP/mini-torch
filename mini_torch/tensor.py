import numpy as np 
import mini_torch.operation as ops 

def convert_to_tensor(obj, requires_grad=False):
    if not isinstance(obj, Tensor):
        obj = Tensor(obj, requires_grad=requires_grad)
    return obj

class Tensor():
    def __init__(self, values, requires_grad=False, dependencies=[], dtype=None):
        self._values = np.asarray(values, dtype)
        
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
        self._values = np.asarray(new_values)
        self.grad = None
    
    @property
    def shape(self):
        return self._values.shape
    
    def __repr__(self):
        return "Tensor(shape=%s, requires_grad=%s)" % (
            self.shape, self.requires_grad)

    def __gt__(self, other):
        return self.values > convert_to_tensor(other).values
    
    def __lt__(self, other):
        return self.values < convert_to_tensor(other).values
    
    def __ge__(self, other):
        return self.values >= convert_to_tensor(other).values
    
    def __le__(self, other):
        return self.values <= convert_to_tensor(other).values

    def __add__(self, other):
        return ops.add_(self, convert_to_tensor(other))
    
    def __radd__(self, other):
        return ops.add_(convert_to_tensor(other), self)

    def __iadd__(self, other):
        self.values = self.values + convert_to_tensor(other).values
        return self

    def __sub__(self, other):
        return ops.sub_(self, convert_to_tensor(other))

    def __rsub__(self, other):
        return ops.sub_(convert_to_tensor(other), self)

    def __isub__(self, other):
        self.values = self.values - convert_to_tensor(other).values
        return self

    def __mul__(self, other):
        return ops.mul_(self, convert_to_tensor(other))

    def __rmul__(self, other):
        return ops.mul_(convert_to_tensor(other), self)

    def __imul__(self, other):
        self.values = self.values * convert_to_tensor(other).values
        return self

    def __truediv__(self, other):
        return ops.div_(self, convert_to_tensor(other))

    def __rtruediv__(self, other):
        return ops.div_(convert_to_tensor(other), self)

    def __itruediv__(self, other):
        self.values = self.values / convert_to_tensor(other).values
        return self

    def __neg__(self):
        return ops.neg_(self)

    def __getitem__(self, key):
        return ops.getitem_(self, key)

    def __pow__(self, other):
        return ops.pow_(self, convert_to_tensor(other))

    def __rpow__(self, other):
        return ops.pow_(convert_to_tensor(other), self)

    def __ipow__(self, other):
        self.values = self.values ** convert_to_tensor(other).values
        return self

    def __matmul__(self, other):
        return ops.matmul_(self, convert_to_tensor(other))

    def __rmatmul__(self, other):
        return ops.matmul_(convert_to_tensor(other), self)

    def __imatmul__(self, other):
        self.values = self.values @ convert_to_tensor(other).values
        return self
    
    def __len__(self):
        return len(self.values)

    def transpose(self, axes=None):
        return ops.transpose_(self, axes=axes)

    def log(self):
        return ops.log_(self)
    
    def sum(self, axis=None):
        return ops.sum_(self, axis=axis)

    @property
    def T(self):
        return ops.transpose_(self, axes=None)

    def zero_grad(self):
        self.grad = np.zeros(self.shape)
    
    def backward(self, grad=None):
        assert self.requires_grad, "Run backward() on a non-requires-grad tensor!"

        gradient = 1.0 if grad is None else grad
        self.grad += np.array(gradient) #梯度累积

        for dependency in self.dependencies:
            grad_for_dep = dependency["grad_func"](gradient)
            dependency["tensor"].backward(grad_for_dep)
    