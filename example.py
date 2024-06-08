#!/usr/bin/env python

from src.scalar import Scalar
from src.graph import Graph

# Manual Backpropagation

# a = Scalar(2, label='a')
# b = Scalar(-3, label='b')
# c = Scalar(10, label='c')
# f = Scalar(-2, label='f')
#
# d = a * b; d.label = 'd'
# e = d + c; e.label = 'e'
# L = e * f; L.label = 'L'
#
# print(f'L before gradient descent: {L.data}')
#
# L.backward()
#
# g = Graph(L)
#
# for x in [a, b, c, f]:
#     x.data += 0.01 * x.grad
#
# d = a * b
# e = d + c
# L = e * f
#
# print(f'L after gradient descent: {L.data}')
# g.show()

# Neuron

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
print(o)

o.zero_grad()
o.backward()

Graph(o).show()

e = 2 * L
f = e.exp()
a = f - 1
b = f + 1
print(a, b)
o = a / b
print(o)

o.zero_grad()
o.backward()

Graph(o).show()
