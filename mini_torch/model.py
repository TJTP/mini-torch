"""
模型
包含权重更新器, 网络, 损失函数的一个完整网络模型
"""

import dill as pickle

class Model():
    def __init__(self, net, loss_layer, optimizer):
        self.net = net 
        self.loss_layer = loss_layer 
        self.optimizer = optimizer
    
    def forward(self, inputs):
        """
        前向传播
        """
        return self.net.forward(inputs)
    
    def step(self):
        """
        更新网络参数
        """
        params_each_layer = self.net.get_net_params()
        grads_each_layer = []
        for params in params_each_layer:
            grads = {}
            for key in params.keys():
                grads[key] = params[key].grad 
            grads_each_layer.append(grads)
        
        steps_each_layer = self.optimizer.get_step(grads_each_layer)

        for steps, params in zip(steps_each_layer, params_each_layer):
            for key in params.keys():
                assert params[key].shape == steps[key].shape 
                params[key] += steps[key]
        
    def zero_grad(self):
        params_each_layer = self.net.get_net_params()
        for params in params_each_layer:
            for values in params.values():
                if values is not None:
                    values.zero_grad()
    
    # 保存或者加载模型, 加载 (load) 是一个类方法
    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.net, file, -1)
        file.close()
    
    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            net = pickle.load(file)
        file.close()
        return net
        
