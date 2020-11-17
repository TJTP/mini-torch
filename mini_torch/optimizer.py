"""
更新梯度的方式
目前只实现了随机梯度下降法 (SGD)
"""
class SGD():
    def __init__(self, learning_rate, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay
    
    def get_step(self, grads_each_layer):
        steps_each_layer = []
        for grads in grads_each_layer:
            for key in grads.keys():
                grads[key] = -grads[key] * self.lr
            steps_each_layer.append(grads)

        return steps_each_layer