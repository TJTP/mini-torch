"""
Tensor类及其重载的操作
"""
import numpy as np 

class DependencyNode():
    def __init__(self, tensor, grad_func):
        self.tensor = tensor
        self.grad_func = grad_func

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
    def values(self, new_val):
        self._values = np.asarray(new_val)
        self.grad = None
    
    @property
    def shape(self):
        return self._values.shape

    @property
    def T(self):
        return self.transpose(axes=None)

    def zero_grad(self):
        self.grad = np.zeros(self.shape)
    
    def backward(self, grad=None):
        """
        反向传播梯度
        """
        assert self.requires_grad == True

        gradient = 1.0 if grad is None else grad
        # 从求导开始到自身的梯度
        self.grad += np.array(gradient) 

        for dependency in self.dependencies:
            dep_grad = dependency.grad_func(gradient)
            dependency.tensor.backward(dep_grad)
    
    # 以下为操作符的重载
    def __gt__(self, obj):
        return self.values > convert_to_tensor(obj).values
    
    def __lt__(self, obj):
        return self.values < convert_to_tensor(obj).values
    
    def __ge__(self, obj):
        return self.values >= convert_to_tensor(obj).values
    
    def __le__(self, obj):
        return self.values <= convert_to_tensor(obj).values

    def __add__(self, obj):
        return add_(self, convert_to_tensor(obj))
    
    def __radd__(self, obj):
        return add_(convert_to_tensor(obj), self)

    def __iadd__(self, obj):
        self.values = self.values + convert_to_tensor(obj).values
        return self

    def __sub__(self, obj):
        return sub_(self, convert_to_tensor(obj))

    def __rsub__(self, obj):
        return sub_(convert_to_tensor(obj), self)

    def __isub__(self, obj):
        self.values = self.values - convert_to_tensor(obj).values
        return self

    def __mul__(self, obj):
        return mul_(self, convert_to_tensor(obj))

    def __rmul__(self, obj):
        return mul_(convert_to_tensor(obj), self)

    def __imul__(self, obj):
        self.values = self.values * convert_to_tensor(obj).values
        return self

    def __truediv__(self, obj):
        return div_(self, convert_to_tensor(obj))

    def __rtruediv__(self, obj):
        return div_(convert_to_tensor(obj), self)

    def __itruediv__(self, obj):
        self.values = self.values / convert_to_tensor(obj).values
        return self

    def __neg__(self):
        return neg_(self)

    def __getitem__(self, key):
        return getitem_(self, key)

    def __pow__(self, obj):
        return pow_(self, convert_to_tensor(obj))

    def __rpow__(self, obj):
        return pow_(convert_to_tensor(obj), self)

    def __ipow__(self, obj):
        self.values = self.values ** convert_to_tensor(obj).values
        return self

    def __matmul__(self, obj):
        return matmul_(self, convert_to_tensor(obj))

    def __rmatmul__(self, obj):
        return matmul_(convert_to_tensor(obj), self)

    def __imatmul__(self, obj):
        self.values = self.values @ convert_to_tensor(obj).values
        return self
    
    def __len__(self):
        return len(self.values)

    # 以下为对numpy中方法的重载
    def transpose(self, axes=None):
        return transpose_(self, axes=axes)

    def log(self):
        return log_(self)
    
    def sum(self, axis=None):
        return sum_(self, axis=axis)

    
        
def convert_to_tensor(obj, requires_grad=False):
    """
    将一个数或者numpy数组转化为Tensor
    """
    if not isinstance(obj, Tensor):
        obj = Tensor(obj, requires_grad=requires_grad)
    return obj
#==========================================================================================   
# utils
def create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1, grad_func_ts2):
    """
    两个操作数形成的tensor (一个计算图上的结点)
    """
    requires_grad = tensor1.requires_grad or tensor2.requires_grad

    dependencies = []
    if tensor1.requires_grad:
        dependencies.append(DependencyNode(tensor=tensor1, grad_func=grad_func_ts1))
    if tensor2.requires_grad:
        dependencies.append(DependencyNode(tensor=tensor2, grad_func=grad_func_ts2))
    
    return Tensor(values, requires_grad, dependencies)

def create_unary_op_tensor(values, tensor1, grad_func_ts1):
    """
    一个操作数形成的tensor (一个计算图上的结点)
    """
    dependencies = []
    if tensor1.requires_grad:
        dependencies.append(DependencyNode(tensor=tensor1, grad_func=grad_func_ts1))
    return Tensor(values, tensor1.requires_grad, dependencies)

def avoid_broadcasting(grad, tensor):
    """
    防止因为broadcasting引起的矩阵尺寸变化, 进而导致传递出现问题
    """
    nidm = grad.ndim
    for _ in range(nidm - tensor.values.ndim):
        # 将grad的维度降到与tensor1的维度一致, 防止出现broadcasting
        grad = grad.sum(axis=0) 
    for i, dim in enumerate(tensor.shape):
        if dim == 1:
            # 如果tensor的某一维数值为1, 则grad按该维相加, 但是保持维度特性. 
            # 也是防止出现broadcasting
            grad = grad.sum(axis=i, keepdims=True)
    return grad    

#==========================================================================================
# operations
def add_(tensor1, tensor2):
    values = tensor1.values + tensor2.values
    
    def grad_func_ts1(grad):
        grad = grad * 1.0
        return avoid_broadcasting(grad, tensor1)
    
    def grad_func_ts2(grad):
        grad = grad * 1.0
        return avoid_broadcasting(grad, tensor2)
    
    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1, 
                                    grad_func_ts2)

def sub_(tensor1, tensor2):
    return tensor1 + (-tensor2)

def mul_(tensor1, tensor2):
    values = tensor1.values * tensor2.values

    def grad_func_ts1(grad):
        grad = grad * tensor2.values #不能写成 grad *= tensor2.values
        return avoid_broadcasting(grad, tensor1)

    def grad_func_ts2(grad):
        grad = grad * tensor1.values
        return avoid_broadcasting(grad, tensor2)
    
    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1,
                                    grad_func_ts2)

def div_(tensor1, tensor2):
    values = tensor1.values / tensor2.values

    def grad_func_ts1(grad):
        grad = grad / tensor2.values
        return avoid_broadcasting(grad, tensor1)
    
    def grad_func_ts2(grad):
        grad = -grad * tensor1.values / tensor2.values ** 2
        return avoid_broadcasting(grad, tensor2)
    
    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1,
                                    grad_func_ts2)
                
def pow_(tensor1, tensor2):
    values = tensor1.values ** tensor2.values

    def grad_func_ts1(grad):
        grad = grad * tensor2.values * tensor1.values ** (tensor2.values - 1)
        return avoid_broadcasting(grad, tensor1)
    
    def grad_func_ts2(grad):
        grad = grad * np.log(tensor1) * values
        return avoid_broadcasting(grad, tensor2)

    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1,
                                    grad_func_ts2)
    
def matmul_(tensor1, tensor2):
    values = tensor1.values @ tensor2.values

    def grad_func_ts1(grad):
        return grad @ tensor2.values.T 
    
    def grad_func_ts2(grad):
        return tensor1.values.T @ grad 
    
    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1,
                                    grad_func_ts2)

def neg_(tensor1):
    values = -tensor1.values

    def grad_func_ts1(grad):
        return -grad 
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def exp_(tensor1):
    values = np.exp(tensor1.values)

    def grad_func_ts1(grad):
        return grad * values

    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def log_(tensor1):
    values = np.log(tensor1.values)

    def grad_func_ts1(grad):
        return grad / tensor1.values
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def transpose_(tensor1, axes=None):
    values = tensor1.values.transpose(axes)

    if axes is None:
        axes = reversed(range(tensor1.values.ndim))
    axes = list(axes)

    def grad_func_ts1(grad):
        return grad.transpose(np.argsort(axes))

    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def sum_(tensor1, axis=None):
    values = tensor1.values.sum(axis=axis)
    
    if not axis is None:
        repeat_num = tensor1.values.shape[axis]
    
    def grad_func_ts1(grad):
        if axis is not None:
            grad = np.expand_dims(grad, axis)
            grad = np.repeat(grad, repeat_num, axis)
        else:
            grad = grad * np.ones_like(tensor1.values)
        return grad 
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def clip_(tensor1, low=None, high=None):
    values = tensor1.values.clip(low, high)

    mask = np.ones(tensor1.shape, dtype=bool)
    if low is not None:
        mask &= (tensor1.values >= low)
    if high is not None:
        mask &= (tensor1.values <= high)

    def grad_func_ts1(grad):
        return grad * mask 
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def getitem_(tensor1, idx):
    values = tensor1.values[idx]

    def grad_func_ts1(grad):
        recover_grad = np.zeros_like(tensor1.values)
        recover_grad[idx] = grad 
        return recover_grad
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

#==========================================================================================
# 在tensor.py 之外的文件中调用时使用的wrapper_function
def exp(obj, requires_grad=False):
    return exp_(convert_to_tensor(obj, requires_grad))

def clip(obj, low=None, high=None, requires_grad=False):
    return (clip_(convert_to_tensor(obj), low, high))

def pow(obj1, obj2, requires_grad=False):
    return pow_(convert_to_tensor(obj1, requires_grad), convert_to_tensor(obj2, requires_grad))