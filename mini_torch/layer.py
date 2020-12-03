"""
网络层
目前只实现了全连接层
"""
from mini_torch.initializer import HeInitializer, ConstInitializer

class DenseLayer:
    def __init__(self, cur_layer_num, last_layer_num=None, 
                 w_initializer=HeInitializer(), b_initializer=ConstInitializer(), show=False):
        
        self.name = "Dense Layer"
        self._initializer = {"w": w_initializer, "b": b_initializer}
        self.params = {"w":None, "b":None}
        self._shapes = {"w": [last_layer_num, cur_layer_num], "b": [1, cur_layer_num]}
        self._is_init = False
        self.shape = None 
        self._cur_layer_num = cur_layer_num

        if last_layer_num is not None:
            self._init_params(last_layer_num)

        self._inputs = None
        self.show = show             
    
    def _init_params(self, last_layer_num):
        # 通常在训练开始后第一轮时, 才会真正初始化层的参数
        self.shape = (last_layer_num, self._cur_layer_num)
        
        self._shapes["w"][0] = last_layer_num
        self.params["w"] = self._initializer["w"].init(self._shapes["w"])
        self.params["b"] = self._initializer["b"].init(self._shapes["b"])

        # 将梯度置零
        self.params["w"].zero_grad()
        self.params["b"].zero_grad()
        self._is_init = True
    
    def forward(self, inputs):
        if not self._is_init:
            self._init_params(inputs.shape[1])
        
        self._inputs = inputs
        output = self._inputs @ self.params["w"] + self.params["b"]
        if self.show:
            print('<'+self.name+'>')
            print(output.values)
        return output