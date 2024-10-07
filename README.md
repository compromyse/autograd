# autograd
An simple implementation of autograd / backpropagation.

All you need to run a simple neural network using autograd is the following code:

The code defines a data set `X`, expected output (or ground truth) `y`. It then trains the neural network by performing backward propagation (`.backward()`), then applies the calculated gradients through `.optimise()` along with a learning rate of `0.01`.

```py
from src.nn import MLP
from src.loss import mse

X = [
    [ 0.0, 1.0, 2.0 ],
    [ 2.0, 1.0, 0.0 ],
    [ 2.0, 2.0, 2.0 ],
    [ 3.0, 3.0, 3.0 ]
]

y = [ 1.0, -1.0, 1.0, -1.0 ]
n = MLP(3, [ 4, 4, 1 ])

for i in range(400):
    pred = [ n(x) for x in X ]
    loss = mse(y, pred)
    loss.zero_grad()
    loss.backward()
    n.optimise(0.01)

print(pred)
```
