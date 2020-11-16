from numpy.testing._private.utils import assert_string_equal


class Network():
    def __init__(self, layers):
        self._layers = layers
        self._is_training = True
    
    def forward(self, inputs):
        for layer in self._layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def set_status(self, is_training):
        for layer in self._layers:
            layer.set_status(is_training)
        self._is_training = is_training
    
    def set_params(self, params):
        for i, layer in enumerate(self._layers):
            assert layer.params.keys() == params[i].keys()
            for key in layer.params.keys():
                assert layer.params[key].shape == params[i][key].shape
                layer.params[key] = params[i][key]
    
    def get_params(self):
        return [layer.params for layer in self._layers]
