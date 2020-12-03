"""
网络
由网络层(layer)组成的具有前向传播的功能的网络
"""

class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def get_net_params(self):
        return [layer.params for layer in self.layers]
    
    def set_show(self, layer_idx):
        self.layers[layer_idx].show = True
