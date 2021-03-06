#!/usr/bin/python

import theano
import theano.tensor as T
import numpy
import random
import re
from itertools import izip 

# random function
def myRand(mn, mx):
   return mn + (random.random() * (mx - mn))

# raw data
raw_data = []

# batch size and number
batch_size = 10
batch_num = 1000

# learning rate
mu = 1.0

# neuron variable declaration
x     = T.matrix("input"    , dtype="float32")
y_hat = T.matrix("reference", dtype="float32")
w1    = theano.shared(numpy.matrix([[myRand(-0.5 , 0.5 ) for j in range(2)  ] for i in range(50)], dtype="float32"))
w2    = theano.shared(numpy.matrix([[myRand(-0.05, 0.05) for j in range(50)] for i in range(50)], dtype="float32"))
w3    = theano.shared(numpy.matrix([[myRand(-0.05, 0.05) for j in range(50)] for i in range(1)  ], dtype="float32"))
b1    = theano.shared(numpy.array([myRand(-0.5, 0.5) for i in range(50)], dtype="float32"))
b2    = theano.shared(numpy.array([myRand(-0.5, 0.5) for i in range(50)], dtype="float32"))
b3    = theano.shared(numpy.array([myRand(-0.5, 0.5) for i in range(1)  ], dtype="float32"))
parameters = [w1, w2, w3, b1, b2, b3]

z1 = T.dot(w1,  x) + b1.dimshuffle(0, 'x')
a1 = 1 / (1 + T.exp(-z1))
z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
a2 = 1 / (1 + T.exp(-z2))
z3 = T.dot(w3, a2) + b3.dimshuffle(0, 'x')
y  = 1 / (1 + T.exp(-z3))
prediction = y > 0.5

# read raw data
def readRawData():
  global raw_data
  f = open("miku.txt", "r")
  while True:
    s = f.readline()
    if s == "": break
    tokens = re.findall(r'[-.0-9]+', s)
    nums = [numpy.float32(x) for x in tokens]
    raw_data += [nums]
  f.close()

def make_batch(data, size, num):
  data_size = len(data)
  X_ret = []
  Y_ret = []
  for i in range(num):
    X_batch = [[], []]
    Y_batch = [[]]
    for j in range(size):
      # random select
      idx = int(random.random() * data_size)
      X_batch[0] += [data[idx][0]]
      X_batch[1] += [data[idx][1]]
      Y_batch[0] += [data[idx][2]]

    X_ret += [X_batch]
    Y_ret += [Y_batch]
  return X_ret, Y_ret

# update function
def updateFunc(param, grad):
  global mu
  param_updates = [(p, p - numpy.float32(mu) * g) for p, g in izip(param, grad)]
  return param_updates

# cost function
cost = T.sum((y - y_hat) ** 2) / batch_size

# gradient function
gradients = T.grad(cost, parameters)

# training function
train = theano.function(
    inputs = [x, y_hat],
    updates = updateFunc(parameters, gradients),
    outputs = cost
)

# testing function
test = theano.function(
    inputs = [x],
    outputs = prediction
)

def main():
  global raw_data, batch_size, batch_num
  global mu
  readRawData()
  mu = 1
  for i in range(200):
    cost = 0
    if i == 60: mu = 0.01
    if i == 140: mu = 0.0001
    X_batch, Y_hat_batch = make_batch(raw_data, batch_size, batch_num)
    for j in range(batch_num):
      cost += train(X_batch[j], Y_hat_batch[j])
    cost /= batch_num
    print i, " cost: ", cost

  result = test([[x[0] for x in raw_data], [x[1] for x in raw_data]])
  correct = 0
  wrong = 0
  f = open("miku.result", "w+")
  for i in range(len(raw_data)):
    f.write("%.2f %.2f %d\n" % (raw_data[i][0], raw_data[i][1], result[0][i]))
    if result[0][i] == raw_data[i][2]:
      correct += 1
    else:
      wrong += 1

  print
  print "correct: ", correct
  print "wrong: ", wrong

if __name__ == "__main__":
   main()
