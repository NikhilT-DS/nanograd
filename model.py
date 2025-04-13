from engine import ValueTensor
import numpy as np


class Linear:
    def __init__(self, input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))  # Xavier init
        self.w = ValueTensor(np.random.uniform(-limit, limit, size=(input_size, output_size)))
        self.b = ValueTensor(np.zeros((1, output_size)))
        self.params = [self.w, self.b]
    def __call__(self, x):
        return x @ self.w + self.b
    def __repr__(self):
        return f"Linear Layer of shape {self.w.shape}"
    def parameters(self):
        return self.params
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
    

class MLP:
    def __init__(self, layer_sizes, activation= "relu"):
        """
        expects the layer sizes as list of integers 
        For e.g. [input size, 32, 64, output size]
        
        """
        
        self.layers = []
        self.activations = {
            "relu": lambda x: x.relu(),
            "tanh": lambda x: x.tanh(),
            "sigmoid": lambda x: x.sigmoid(),
            None: lambda x: x
        }

        assert activation in self.activations, f"Unsupported activation: {activation}"
        self.act_fn = self.activations[activation]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
        
    
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act_fn(x)
        return x
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    def __repr__(self):
        return f"MLP"

