import math

class Scalar:
    def __init__(self, data, _children=(), _op='', label='') -> None:
        self.label = label

        self.data = float(data)
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op
        
        self._backward = lambda: None
    
    def __repr__(self) -> str:
        return f'Scalar({self.label}: {self.data})'

    def __add__(self, y):
        result = Scalar(self.data + y.data, (self, y), _op='+')

        def _backward():
            self.grad = result.grad
            y.grad = result.grad

        self._backward = _backward

        return result

    def __mul__(self, y):
        result = Scalar(self.data * y.data, (self, y), _op='*')

        def _backward():
            self.grad = y.data * result.grad
            y.grad = self.data * result.grad

        self._backward = _backward

        return result

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        result = Scalar(t, (self, ), 'tanh')

        def _backward():
            self.grad = (1 - (t ** 2)) * result.grad

        self._backward = _backward

        return result

    def build_children(self):
        result = []

        result.append(self)
        for child in self._prev:
            result += child.build_children()

        return result

    def backward(self):
        self.grad = 1.0
        children = self.build_children()

        for child in children:
            child._backward()
