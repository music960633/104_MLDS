#!/usr/bin/python

import theano
import theano.tensor as T
import numpy
import random
import re
from itertools import izip 

# raw data
raw_data = [[1,2,3]] * 100

# batch size and number
batch_size = 5
batch_num = 4

# neuron variable declaration
x     = T.matrix("input")
y_hat = T.matrix("reference")
w1    = theano.shared(numpy.matrix([[0.5, 0.5]] * 1000))
w2    = theano.shared(numpy.matrix([[0.5]*1000] * 1000))
w3    = theano.shared(numpy.matrix([[0.5]*1000] * 1   ))
b1    = theano.shared(numpy.array([0.0] * 1000))
b2    = theano.shared(numpy.array([0.0] * 1000))
b3    = theano.shared(numpy.array([0.0] * 1   ))
parameters = [w1, w2, w3, b1, b2, b3]

z1 = T.dot(w1, x) + b1.dimshuffle(0, 'x')
a1 = 1 / (1 + T.exp(-z1))
z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
a2 = 1 / (1 + T.exp(-z2))
z3 = T.dot(w3, a2) + b3.dimshuffle(0, 'x')
y  = 1 / (1 + T.exp(-z3))

add = T.sum(y)

def make_batch(size, num):
  print size, num
  global raw_data
  data_size = len(raw_data)
  X_ret = []
  Y_ret = []
  for i in range(num):
    X_batch = [[], []]
    Y_batch = []
    for j in range(size):
      idx = int(random.random() * data_size)
      X_batch[0] += [raw_data[idx][0]]
      X_batch[1] += [raw_data[idx][1]]
      Y_batch    += [raw_data[idx][2]]
    X_ret += [X_batch]
    Y_ret += [Y_batch]
  return X_ret, Y_ret

# training function
train = theano.function(
    inputs = [x],
    outputs = add
)

def main():
  global batch_size, batch_num
  for i in range(100):
    X_batch, Y_hat_batch = make_batch(batch_size, batch_num)
    print X_batch, Y_hat_batch
    for j in range(batch_num):
      print train(X_batch[j])

if __name__ == "__main__":
   main()
