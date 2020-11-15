import numpy as np 
from mini_torch.tensor import Tensor


class XavierInitializer():
    """
    Xavier 初始化
    当用Sigmoid函数, 或双曲正切函数作为激活函数时, 用此方法初始化权重
    """
    def __init__(self, gamma=4.0, type="Normal"):
        self._gamma = gamma
        self._type = type

    def init(self, shape):
        if self._type == "Normal":
            last_layer_num, cur_layer_num = shape[0], shape[1]
            std = self._gamma * np.sqrt(2.0 / (last_layer_num + cur_layer_num))
            return Tensor(np.random.normal(loc=0.0, scale=std, size=shape), 
                          requires_grad=True,
                          dtype=np.float32)
        elif self._type == "Uniform":
            last_layer_num, cur_layer_num = shape[0], shape[1]
            a = self._gamma * np.sqrt(6.0 / (last_layer_num + cur_layer_num))
            return Tensor(np.random.uniform(low=-a, high=a, size = shape),
                          requires_grad=True,
                          dtype=np.float32)
        assert "Invalid type!"


class HeInitializer():
    """
    Kaiming He 初始化
    当用ReLu作为激活函数时, 用此方法初始化权重
    """
    def __init__(self, gamma=1.0, type="Normal"):
        self._gamma = gamma
        self._type = type
    
    def init(self, shape):
        if self._type == "Normal":
            last_layer_num = shape[0]
            std = self._gamma * np.sqrt(2.0 / last_layer_num)
            return Tensor(np.random.normal(loc=0.0, scale=std, size=shape), 
                          requires_grad=True,
                          dtype=np.float32)
        elif type == "Uniform":
            last_layer_num = shape[0]
            a = self._gamma * np.sqrt(6.0 / last_layer_num)
            return Tensor(np.random.uniform(low=-a, high=a, size = shape),
                          requires_grad=True,
                          dtype=np.float32)
        assert "Invalid type!"
    
class ConstInitializer():
    def __init__(self, value=0.0):
        self._value = value
    
    def init(self, shape):
        return Tensor(np.full(shape, self._value), 
                      requires_grad=True,
                      dtype=np.float32)

    
    
