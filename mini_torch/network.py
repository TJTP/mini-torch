"""
网络
由网络层(layer)组成的具有前向传播的功能的网络
"""

class Network():
    def __init__(self, layers):
        self._layers = layers
    
    def forward(self, inputs):
        for layer in self._layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def get_net_params(self):
        return [layer.params for layer in self._layers]
