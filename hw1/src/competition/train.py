#!/usr/bin/python

import theano
import theano.tensor as T
import numpy
import random
import re
from itertools import izip
read = __import__("read")

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

#lookup table
map_48_39 = read.get_map_48_39()
map_48_idx = {}

def setPhoneIdx():
  global map_48_39, map_48_idx
  temp = map_48_39.items()
  for i in range(48):
    map_48_idx[temp[i][0]] = i

# neuron variable declaration
x     = T.matrix("input"    , dtype="float32")
y_hat = T.matrix("reference", dtype="float32")
w1    = theano.shared(numpy.matrix([[myRand(-0.5 , 0.5 ) for j in range(69)  ] for i in range(100)], dtype="float32"))
w2    = theano.shared(numpy.matrix([[myRand(-0.05, 0.05) for j in range(100)] for i in range(100)], dtype="float32"))
w3    = theano.shared(numpy.matrix([[myRand(-0.05, 0.05) for j in range(100)] for i in range(100)], dtype="float32"))
w4    = theano.shared(numpy.matrix([[myRand(-0.05, 0.05) for j in range(100)] for i in range(1)  ], dtype="float32"))
b1    = theano.shared(numpy.array([myRand(-0.5, 0.5) for i in range(100)], dtype="float32"))
b2    = theano.shared(numpy.array([myRand(-0.5, 0.5) for i in range(100)], dtype="float32"))
b3    = theano.shared(numpy.array([myRand(-0.5, 0.5) for i in range(100)], dtype="float32"))
b4    = theano.shared(numpy.array([myRand(-0.5, 0.5) for i in range(1)  ], dtype="float32"))
parameters = [w1, w2, w3, w4, b1, b2, b3, b4]

z1 = T.dot(w1,  x) + b1.dimshuffle(0, 'x')
a1 = 1 / (1 + T.exp(-z1))
z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
a2 = 1 / (1 + T.exp(-z2))
z3 = T.dot(w3, a2) + b3.dimshuffle(0, 'x')
a3 = 1 / (1 + T.exp(-z3))
z4 = T.dot(w4, a3) + b4.dimshuffle(0, 'x')
y  = 1 / (1 + T.exp(-z4))
prediction = y > 0.5

def make_batch(data, size, num, ans):
  data_size = len(data)
  X_ret = []
  Y_ret = []
  for i in range(num):
    X_batch = [[] for t in range(69)]
    Y_batch = [[0 for u in range(size)] for t in range(48)]
    print Y_batch
    for j in range(size):
      # random select
      idx = int(random.random() * data_size)
      for k in range(69):
        X_batch[k] += [data[idx][1][k]]
      phoneIdx = map_48_idx[ans[data[idx][0]]]
      Y_batch[phoneIdx][j] = 1
    X_ret += [X_batch]
    Y_ret += [Y_batch]
  print X_ret, Y_ret
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
  raw_data = read.get_fbank()
  raw_data = raw_data.items()
  ans = read.get_map_inst_48()
  setPhoneIdx()
  mu = 1
  for i in range(100):
    cost = 0
    if i == 50: mu = 0.01
    X_batch, Y_hat_batch = make_batch(raw_data, batch_size, batch_num, ans)
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
