import random
from src.micrograd import Value

class Neuron:
    def __init__(self, input_dim):
       self.w =  [Value(random.uniform(-1, 1)) for _ in range(input_dim)]
       self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        o = sum((w * x for w, x in zip(self.w, x)), self.b)
        return o.tanh()
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, input_dim, output_dim):
       self.neurons = [Neuron(input_dim) for _ in range(output_dim)] 
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [w for neuron in self.neurons for w in neuron.parameters()]

class NeuralNetwork:
    def __init__(self, input_dim, layers):
        sizes = [input_dim] + layers
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(layers))]
        
    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(x)
        return y
    
    def parameters(self):
        return [w for layer in self.layers for w in layer.parameters()]
    
    def zero_grad(self):
        params = self.parameters()
        
        for p in params:
            p.zero_grad()