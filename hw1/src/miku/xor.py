#!/usr/bin/python

import theano
import theano.tensor as T
import numpy
import random
import re
from itertools import izip 

# raw data
raw_data = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

# batch size and number
batch_size = 4
batch_num = 1

# neuron variable declaration
x     = T.matrix("input", dtype="float32")
y_hat = T.matrix("reference", dtype="float32")
w1    = theano.shared(numpy.matrix([[random.random() for j in range(2)] for i in range(2)], dtype="float32"))
w2    = theano.shared(numpy.matrix([[random.random() for j in range(2)] for i in range(1)], dtype="float32"))
b1    = theano.shared(numpy.array([random.random() for i in range(2)], dtype="float32"))
b2    = theano.shared(numpy.array([random.random() for i in range(1)], dtype="float32"))
parameters = [w1, w2, b1, b2]

z1 = T.dot(w1, x) + b1.dimshuffle(0, 'x')
a1 = 1 / (1 + T.exp(-z1))
z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
y  = 1 / (1 + T.exp(-z2))

def make_batch(data, size, num):
  data_size = len(data)
  X_ret = []
  Y_ret = []
  for i in range(num):
    X_batch = [[], []]
    Y_batch = [[]]
    for j in range(size):
      X_batch[0] += [data[j][0]]
      X_batch[1] += [data[j][1]]
      Y_batch[0] += [data[j][2]]

    X_ret += [X_batch]
    Y_ret += [Y_batch]
  return X_ret, Y_ret

# update function
def updateFunc(param, grad):
  mu = numpy.float32(2)
  param_updates = [(p, p - mu * g) for p, g in izip(param, grad)]
  return param_updates

# cost function
cost = T.sum((y - y_hat) ** 2) / 4 # batch_size

# gradient function
gradients = T.grad(cost, parameters)

# training function
train = theano.function(
    inputs = [x, y_hat],
    updates = updateFunc(parameters, gradients),
    outputs = y
)

# testing function
test = theano.function(
    inputs = [x],
    outputs = y
)

def main():
  global raw_data,  batch_size, batch_num
  for i in range(1000):
    cost = 0
    X_batch, Y_hat_batch = make_batch(raw_data, batch_size, batch_num)
    for j in range(batch_num):
      print X_batch[j]
      print Y_hat_batch[j]
      print train(X_batch[j], Y_hat_batch[j])
      print "w1:", w1.get_value()
      print "b1:", b1.get_value()
      print "w2:", w2.get_value()
      print "b2:", b2.get_value()

if __name__ == "__main__":
   main()
