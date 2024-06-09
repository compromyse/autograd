#!/usr/bin/env python

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

pred = [ n(x) for x in X ]
print(pred)

for i in range(400):
    pred = [ n(x) for x in X ]
    loss = mse(y, pred)

    loss.zero_grad()
    loss.backward()
    n.optimise(0.01)

    print(loss.data)

print(pred)
