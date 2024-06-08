#!/usr/bin/env python

from src.scalar import Scalar

h = 0.0001

# def one():
#     a = Scalar(2, label='a')
#     b = Scalar(-3, label='b')
#     c = Scalar(10, label='c')
#     f = Scalar(-2, label='f')
#
#     d = a * b; d.label = 'd'
#     e = d + c; e.label = 'e'
#     L = e * f; L.label = 'L'
#
#     return L.data
#
# def two():
#     a = Scalar(2, label='a')
#     b = Scalar(-3, label='b')
#     c = Scalar(10, label='c')
#     f = Scalar(-2, label='f')
#
#     d = a * b; d.label = 'd'
#     d.data += h
#     e = d + c; e.label = 'e'
#     L = e * f; L.label = 'L'
#
#     return L.data
#
# print((two() - one()) / h)

a = Scalar(2, label='a')
b = Scalar(-3, label='b')
c = Scalar(10, label='c')
f = Scalar(-2, label='f')

d = a * b; d.label = 'd'
e = d + c; e.label = 'e'
L = e * f; L.label = 'L'

L.grad = 1.0
e.grad = -2.0
f.grad = 4.0
d.grad = -2.0
c.grad = -2.0

from src.graph import Graph
Graph(L).show()
