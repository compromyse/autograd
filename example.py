#!/usr/bin/env python

from src.value import Value
from src.graph import Graph

a = Value(2, label='a')
b = Value(-3, label='b')
c = Value(10, label='c')
f = Value(-2, label='f')

d = a * b; d.label = 'd'
e = d + c; e.label = 'e'
L = e * f; L.label = 'L'

L.grad = 1.0
e.grad = -2.0
f.grad = 4.0

Graph(L).show()
