#!/usr/bin/env python

from src.scalar import Scalar
from src.graph import Graph

x1 = Scalar(2, label='x1')
x2 = Scalar(0, label='x2')

w1 = Scalar(-3, label='w1')
w2 = Scalar(1, label='w2')

b = Scalar(6.7, label='b')

x1w1 = x1 * w1; x1w1.label = 'x1w1'
x2w2 = x2 * w2; x2w2.label = 'x2w2'

x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'

L = x1w1x2w2 + b; L.label = 'L'

o = L.tanh(); o.label = 'o'

o.zero_grad()
o.backward()

Graph(o).show()

e = 2 * L
f = e.exp()
a = f - 1
b = f + 1
o = a / b

o.zero_grad()
o.backward()

Graph(o).show()
