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
# L.grad = 1.0
# e.grad = -2.0
# f.grad = 4.0
# d.grad = -2.0
# c.grad = -2.0
# a.grad = 6.0
# b.grad = -4.0
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

# o.grad = 1.0
# L.grad = 1 - (o.data ** 2)
# b.grad = L.grad
# x1w1x2w2.grad = L.grad
# x1w1.grad = x1w1x2w2.grad
# x2w2.grad = x1w1x2w2.grad
#
# x1.grad = w1.data * x1w1.grad
# w1.grad = x1.data * x1w1.grad
#
# x2.grad = w2.data * x2w2.grad
# w2.grad = x2.data * x2w2.grad

o.backward()

Graph(o).show()
