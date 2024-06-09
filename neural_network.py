#!/usr/bin/env python

from src.nn import MLP

X = [
    [ 0.0, 0.0, 0.0 ],
    [ 1.0, 1.0, 1.0 ],
    [ 2.0, 2.0, 2.0 ],
    [ 3.0, 3.0, 3.0 ]
]

y = [ 1.0, 2.0, 3.0, 4.0 ] # X + 1

n = MLP(3, [ 4, 4, 1 ])

pred = [ n(x) for x in X ]
print(pred)
