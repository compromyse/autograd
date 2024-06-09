from .scalar import Scalar
import random

class Neuron:
    def __init__(self, n_X):
        self.n_X = n_X
        self.w = [ Scalar(random.uniform(-1, 1)) for _ in range(n_X) ]
        self.b = Scalar(random.uniform(-1, 1))

    def __call__(self, X):
        result = 0

        for wi, Xi in zip(self.w, X):
            result += wi * Xi

        result += self.b

        return result.tanh()

class Layer:
    def __init__(self, n_X, n_y):
        self.neurons = [ Neuron(n_X) for _ in range(n_y) ]

    def __call__(self, X):
        result = [ n(X) for n in self.neurons ]
        return result[0] if len(result) == 1 else result

class MLP:
    def __init__(self, n_X, layers):
        sz = [ n_X ] + layers
        self.layers = [ Layer(sz[i], sz[i + 1]) for i in range(len(layers)) ]

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        
        return X
