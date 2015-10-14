#!/usr/bin/python

import theano
import theano.tensor as T
import numpy
import random
import re
from itertools import izip 

# raw data
data = []

# neuron variable declaration
x = T.vector()
w = theano.shared(numpy.array([1.0, -1.0]))
b = theano.shared(0.0)

z = T.dot(w, x) + b
y = 1 / (1 + T.exp(-z)) # activation function


y_hat = T.scalar()
cost = T.sum((y - y_hat) ** 2)
dw, db = T.grad(cost, [w, b])

# read raw data
def readRawData():
  global data
  f = open("miku.txt", "r")
  while True:
    s = f.readline()
    if s == "": break
    tokens = re.findall(r'[.0-9]+', s)
    nums = [float(x) for x in tokens]
    data += [nums]
  f.close()

# update function
def updateFunc(param, grad):
  mu = 0.3
  param_updates = [(p, p - mu * g) for p, g in izip(param, grad)]
  return param_updates

# neuron function
neuron = theano.function(inputs = [x], outputs = y)

# grdient function
gradient = theano.function(
    inputs = [x, y_hat],
    updates = updateFunc([w, b], [dw, db])
)

def main():
  readRawData()
  for i in range(100):
    x = data[i][0:2]
    y_hat = data[i][2]
    print neuron(x)
    gradient(x, y_hat)
    print w.get_value(), b.get_value()

if __name__ == "__main__":
   main()
