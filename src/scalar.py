class Scalar:
    def __init__(self, data, _children=(), _op='', label='') -> None:
        self.label = label

        self.data = float(data)
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self) -> str:
        return f'Scalar({self.data})'

    def __add__(self, y):
        result = self.data + y.data
        return Scalar(result, (self, y), _op='+')

    def __mul__(self, y):
        result = self.data * y.data
        return Scalar(result, (self, y), _op='*')
