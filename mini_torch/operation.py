import numpy as np

def create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1, grad_func_ts2):
    requires_grad = tensor1.requires_grad or tensor2.requires_grad
    dependencies = []
    if tensor1.requires_grad:
        dependencies.append(dict(tensor=tensor1, grad_func=grad_func_ts1))
    if tensor2.requires_grad:
        dependencies.append(dict(tensor=tensor2, grad_func=grad_func_ts2))
    return tensor1.__class__(values, requires_grad, dependencies)

def create_unary_op_tensor(values, tensor1, grad_func_ts1):
    dependencies = []
    if tensor1.requires_grad:
        dependencies.append(dict(tensor=tensor1, grad_func=grad_func_ts1))
    return tensor1.__class__(values, tensor1.requires_grad, dependencies)

def avoid_broadcasting(grad, tensor):
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

def add_(tensor1, tensor2):
    values = tensor1.values + tensor2.values
    
    # c = a + b
    # dc / da = 1.0
    # dc / db = 1.0
    def grad_func_ts1(grad):
        #grad *= 1.0
        return avoid_broadcasting(grad, tensor1)
    
    def grad_func_ts2(grad):
        #grad *= 1.0
        return avoid_broadcasting(grad, tensor2)
    
    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1, 
                                    grad_func_ts2)

def sub_(tensor1, tensor2):
    return tensor1 + (-tensor2)

def mul_(tensor1, tensor2):
    values = tensor1.values * tensor2.values

    def grad_func_ts1(grad):
        grad *= tensor2.values
        return avoid_broadcasting(grad, tensor1)

    def grad_func_ts2(grad):
        grad *= tensor1.values
        return avoid_broadcasting(grad, tensor2)
    
    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1,
                                    grad_func_ts2)

def div_(tensor1, tensor2):
    values = tensor1.values / tensor2.values

    def grad_func_ts1(grad):
        grad /= tensor2.values
        return avoid_broadcasting(grad, tensor1)
    
    def grad_func_ts2(grad):
        grad = -grad * tensor1.values / tensor2.values ** 2
        return avoid_broadcasting(grad, tensor2)
    
    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1,
                                    grad_func_ts2)
                
def  pow_(tensor1, tensor2):
    values = tensor1.values ** tensor2.values

    def grad_func_ts1(grad):
        grad *= tensor2.values * tensor1.values ** (tensor2.values - 1)
        return avoid_broadcasting(grad, tensor1)
    
    def grad_func_ts2(grad):
        grad *= np.log(tensor1) * values
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

def getitem_(tensor1, idx):
    values = tensor1.values[idx]

    def grad_func_ts1(grad):
        recover_grad = np.zeros_like(tensor1.values)
        recover_grad[idx] = grad 
        return recover_grad
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def sum_(tensor1, axis=None):
    values = tensor1.values.sum(axis=axis)
    
    if not axis is None:
        repeat_num = tensor1.values.shape[axis]
    
    def grad_func_ts1(grad):
        if not axis is None:
            grad = np.expand_dims(grad, axis)
            grad = np.repeat(grad, repeat_num, axis)
        else:
            grad = grad * np.ones_like(tensor1.values)
        return grad 
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)

def clip_(tensor1, low, high):
    values = tensor1.values.clip(low, high)

    mask = np.ones(tensor1.shape, dtype=bool)
    if low is not None:
        mask &= (tensor1.values >= low)
    if high is not None:
        mask &= (tensor1.values <= high)

    def grad_func_ts1(grad):
        return grad * mask 
    
    return create_unary_op_tensor(values, tensor1, grad_func_ts1)
        
#def max_(tensor1, axis=None):
#    values = np.max(tensor1.values, axis=axis)

#    def grad_func_ts1(grad):
#        return grad * (tensor1.values.max(axis=axis, keepdims=True) == tensor1.values)

#def maximum_(tensor1, tensor2):
#    values = np.maximum(tensor1.values, tensor2.values)
#
#    def grad_func_ts1(grad):
#        grad *= (tensor1.values >= tensor2.values)
#        return avoid_broadcasting(grad, tensor1)
#    
#    def grad_func_ts2(grad):
#        grad *= (tensor2.values >= tensor1.values)
#        return avoid_broadcasting(grad, tensor2)
#
#    return create_binary_ops_tensor(values, tensor1, tensor2, grad_func_ts1,
#                                    grad_func_ts2)