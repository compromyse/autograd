import math

class Scalar:
    def __init__(self, data, _children=(), _op='', label='') -> None:
        self.label = label

        self.data = float(data)
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op
        
        self._backward = lambda: None
    
    def __add__(self, y):
        y = y if isinstance(y, Scalar) else Scalar(y)
        result = Scalar(self.data + y.data, (self, y), _op='+')

        def _backward():
            self.grad += result.grad
            y.grad += result.grad

        self._backward = _backward

        return result

    def __mul__(self, y):
        y = y if isinstance(y, Scalar) else Scalar(y)
        result = Scalar(self.data * y.data, (self, y), _op='*')

        def _backward():
            self.grad += y.data * result.grad
            y.grad += self.data * result.grad

        self._backward = _backward

        return result

    def __pow__(self, y):
        assert isinstance(y, (int, float))
        result = Scalar(self.data ** y, (self, ), _op=f'** {y}')

        def _backward():
            self.grad += (y * self.data ** (y - 1)) * result.grad

        self._backward = _backward

        return result

    def exp(self):
        x = self.data
        e = math.exp(x)
        result = Scalar(e, (self, ), 'exp')

        def _backward():
            self.grad += result.data * result.grad

        self._backward = _backward

        return result

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        result = Scalar(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * result.grad

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

    def zero_grad(self):
        self.grad = 0.0
        children = self.build_children()

        for child in children:
            child.grad = 0.0

    def __truediv__(self, y):
        return self * y ** -1

    def __rtruediv__(self, y):
        return self * y ** -1

    def __neg__(self):
        return self * -1

    def __sub__(self, y):
        return self + (-y)

    def __rsub__(self, y):
        return self + (-y)

    def __radd__(self, y):
        return self + y

    def __rmul__(self, y):
        return self * y

    def __repr__(self) -> str:
        return f'Scalar({self.data})'

