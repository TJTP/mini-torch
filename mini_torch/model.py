import pickle

class Model():
    def __init__(self, net, loss_layer, optimizer):
        self.net = net 
        self.loss_layer = loss_layer 
        self.optimizer = optimizer
        
        self._is_training = True
    
    def forward(self, inputs):
        return self.net.forward(inputs)
    
    def step(self):
        params_each_layer = self.net.get_params()
        grads_each_layer = []
        for params in params_each_layer:
            grads = {}
            for key in params.keys():
                grads[key] = params[key].grad 
            grads_each_layer.append(grads)
        
        steps_each_layer = self.optimizer.compute_step(grads_each_layer)

        for steps, params in zip(steps_each_layer, params_each_layer):
            for key in params.keys():
                params[key] += steps[key]
        
    def zero_grad(self):
        params_each_layer = self.net.get_params()
        for param in params_each_layer:
            for values in param.values():
                if values is not None:
                    values.zero_grad()
    
    def set_status(self, is_training):
        self.net.set_status(is_training)
        self._is_training = is_training
    
    
    def save(path, net):
        with open(path, "wb") as file:
            pickle.dump(net, file, 2)
        file.close()
    
    @classmethod
    def load(path):
        with open(path, "wb") as file:
            net = pickle.load(file)
        file.close()
        return net
        
