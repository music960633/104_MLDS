import theano
import theano.tensor as T
import numpy
import random
import re
from itertools import izip

import readdata

# random function
def myRand(mn, mx):
   return mn + (random.random() * (mx - mn))

# raw data
train_inst, train_fbank = readdata.get_train_fbank()
test_inst , test_fbank  = readdata.get_test_fbank()
map_inst_48 = readdata.get_map_inst_48()
map_48_39   = readdata.get_map_48_39()
map_idx_48  = dict(enumerate(map_48_39.keys(), 0))
map_48_idx  = dict(zip(map_idx_48.values(), map_idx_48.keys()))

# batch size and number
batch_size = 10
batch_num = 1000

# learning rate
mu = 1.0

# neuron variable declaration
x     = T.matrix("input"    , dtype="float32")
y_hat = T.matrix("reference", dtype="float32")
w1    = theano.shared(numpy.matrix([[myRand(-0.5, 0.5) for j in range(69) ] for i in range(128)], dtype="float32"))
w2    = theano.shared(numpy.matrix([[myRand(-0.5, 0.5) for j in range(128)] for i in range(128)], dtype="float32"))
w3    = theano.shared(numpy.matrix([[myRand(-0.5, 0.5) for j in range(128)] for i in range(48) ], dtype="float32"))
b1    = theano.shared(numpy.array([myRand(-0.5, 0.5) for i in range(128)], dtype="float32"))
b2    = theano.shared(numpy.array([myRand(-0.5, 0.5) for i in range(128)], dtype="float32"))
b3    = theano.shared(numpy.array([myRand(-0.5, 0.5) for i in range(48) ], dtype="float32"))
parameters = [w1, w2, w3, b1, b2, b3]

z1 = T.dot(w1,  x) + b1.dimshuffle(0, 'x')
a1 = 1 / (1 + T.exp(-z1))
z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
a2 = 1 / (1 + T.exp(-z2))
z3 = T.dot(w3, a2) + b3.dimshuffle(0, 'x')
y  = 1 / (1 + T.exp(-z3))


def make_batch(size, num):
  global train_inst, train_fbank
  global map_inst_48, map_idx_48
  data_size = len(train_inst)
  X_ret = []
  Y_ret = []
  for i in range(num):
    X_batch = [[] for i in range(69)]
    Y_batch = [[] for i in range(48)]
    for j in range(size):
      # random select
      idx = int(random.random() * data_size)
      for k in range(69):
        X_batch[k] += [train_fbank[idx][k]]
      for k in range(48):
        Y_batch[k] += [numpy.float32(1) if map_inst_48[train_inst[idx]] == map_idx_48[k] else numpy.float32(0)]
    X_ret += [X_batch]
    Y_ret += [Y_batch]
  print X_ret, Y_ret
  return X_ret, Y_ret

# temporarily make test = train
def make_test():
  global train_fbank
  test_X = [[] for i in range(69)]
  for i in range(len(train_fbank)):
    for k in range(69):
      test_X[k] += [train_fbank[i][k]]
  return test_X

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
    outputs = y
)

def match(arr):
  global map_idx_48, map_48_39
  idx = 0
  mx = arr[0]
  for i in range(len(arr)):
    if arr[i] > mx:
      idx = i
      mx = arr[i]
  return map_48_39[map_idx_48[idx]]


def run():
  global batch_size, batch_num
  global test_inst
  global mu
  mu = 0.1
  print "start training"
  for i in range(10000):
    cost = 0
    X_batch, Y_hat_batch = make_batch(batch_size, batch_num)
    for j in range(batch_num):
      cost += train(X_batch[j], Y_hat_batch[j])
    cost /= batch_num
    print i, " cost: ", cost

  X_test = make_test()
  result = test(X_test)
  
  f = open("result.csv", "w+")
  f.write("Id,Prediction\n")
  # for i in range(len(test_inst)):
  #   f.write("%s,%s\n" % (test_inst[i], match([result[j][i] for j in range(48)])))
  for i in range(len(train_inst)):
    f.write("%s,%s\n" % (train_inst[i], match([result[j][i] for j in range(48)])))
  f.close()

