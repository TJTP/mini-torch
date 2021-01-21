"""
模型
包含权重更新器, 网络, 损失函数的一个完整网络模型
"""

import dill as pickle

class Model():
    def __init__(self, net, loss_layer, optimizer):
        self._net = net 
        self.loss_layer = loss_layer 
        self._optimizer = optimizer
    
    def forward(self, inputs):
        """
        前向传播
        """
        return self._net.forward(inputs)
    
    def step(self):
        """
        更新网络参数
        """
        params_each_layer = self._net.get_net_params()
        grads_each_layer = []
        for params in params_each_layer:
            grads = {}
            for key in params.keys():
                grads[key] = params[key].grad 
            grads_each_layer.append(grads)
        
        steps_each_layer = self._optimizer.compute_steps_each_layer(grads_each_layer)

        for steps, params in zip(steps_each_layer, params_each_layer):
            for key in params.keys():
                assert params[key].shape == steps[key].shape 
                params[key] += steps[key]
        
    def zero_grad(self):
        params_each_layer = self._net.get_net_params()
        for params in params_each_layer:
            for values in params.values():
                if values is not None:
                    values.zero_grad()
    
    # 保存或者加载模型, 加载 (load) 是一个类方法
    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self._net, file, 2)
        file.close()
    
    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            net = pickle.load(file)
        file.close()
        return net
    
    def get_layer_params(self):
        return self._net.get_net_params()

    def _print_structure(self):
        layerCnt = 0
        print("[Input--->]")
        for layer in self._net.layers:
            if (layer.name[-3:] == "act"):
                print(layer.name+'-'+str(layerCnt), end='\n')
            else:
                print(layer.name+'('+str(layer.shape[1])+')'+'-'+str(layerCnt), end='  ')
            layerCnt += 1
        print("\n[Output--->]")
    
    def set_show(self, layer_idxs):
        for layer_idx in layer_idxs:
            self._net.set_show(layer_idx)
        
    def record_layer_values(self, layer_idxs):
        for layer_idx in layer_idxs:
            self._net.record_layer_values(layer_idx)
    
    def get_layer_values(self, layer_idx):
        return self._net.get_layer_values(layer_idx)
    
    def clean_layer_values(self, layer_idxs):
        for layer_idx in layer_idxs:
            self._net.clean_layer_values(layer_idx)
    
    def print_model(self):
        print("========Model structure========")
        self._print_structure()
        print("Loss function: %s"%(self.loss_layer.name))
        if self._optimizer is not None:
            print("Optimizer: %s"%(self._optimizer.name))
        print("===============================")
