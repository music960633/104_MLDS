import theano
import theano.tensor as T
import numpy as np
import random
import re
import time
from itertools import izip


import readdata

# raw data
#train_inst  = []
#train_fbank = []
test_inst   = []
test_fbank  = []
map_inst_48 = {}
map_48_39   = {}
map_idx_48  = {}
map_48_idx  = {}

# learning rate
mu = 0.1

# parameter
N_HIDDEN = 100
N_INPUT = 48
N_OUTPUT = 48
# neuron variable declaration
x_seq     = T.matrix("input")
y_hat_seq = T.matrix("reference")
Wi   = theano.shared(np.matrix([[random.gauss(0.0, 0.1) for j in range(N_HIDDEN)] for i in range(N_INPUT )]))
Wh   = theano.shared(np.matrix([[random.gauss(0.0, 0.1) for j in range(N_HIDDEN)] for i in range(N_HIDDEN)]))
Wo   = theano.shared(np.matrix([[random.gauss(0.0, 0.1) for j in range(N_OUTPUT)] for i in range(N_HIDDEN)]))
bo   = theano.shared(np.array ([ random.gauss(0.0, 0.1) for i in range(N_OUTPUT)]))
bh   = theano.shared(np.array ([ random.gauss(0.0, 0.1) for i in range(N_HIDDEN)]))
a_0 = theano.shared(np.zeros(N_HIDDEN))
y_0 = theano.shared(np.zeros(N_OUTPUT))
parameters = [Wi, bh, Wo, bo, Wh]

def init():
  print "initializing..."
  global train_inst, train_fbank
  global test_inst, test_fbank
  global map_inst_48, map_48_39
  global map_idx_48, map_48_idx
  global map_48_char
  print "reading testing data"
  test_inst , test_fbank  = readdata.get_test_post()
  # instance name and phone mapping
  print "reading instance name - phone mapping"
  map_inst_48 = readdata.get_map_inst_48()
  map_48_39   = readdata.get_map_48_39()
  map_48_char = readdata.get_map_48_char()
  # phone and index mapping
  print "generating phone - index mapping"
  map_idx_48  = dict(enumerate(map_48_39.keys(), 0))
  map_48_idx  = dict(zip(map_idx_48.values(), map_idx_48.keys()))

def gen_data():
  global  map_inst_48
  train_inst, train_fbank = readdata.get_small_train_data()
  X_seq = []
  Y_hat_seq = []
  i = 0
  size =  len(train_fbank[0])
  while i < size:
    X_seq += [[train_fbank[row][i] for row in range(N_INPUT)]]
    Y_hat_seq += [[(1.0 if map_inst_48[train_inst[i]] == map_idx_48[row] else 0.0) for row in range(N_OUTPUT)]]
    i = i + 1
  return X_seq, Y_hat_seq


# update function
def updateFunc(param, grad):
  global mu
  parameters_updates = [(p,p - mu * g) for p,g in izip(parameters,gradients) ] 
  return parameters_updates

momentum = 0.9
lamb = 0.5
def momentum (param, grad):
  global mu, lamb, momentum, movement
  param_updates = []
  for p, g in zip(param, grad):
    movement = theano.shared(0.)
    movement = (lamb * movement) - mu * g
    param_updates += [(p, p + movement)]
  return param_updates

def sigmoid(z):
  return 1/(1 + T.exp(-z))

def step (x_t, a_tm1, y_tm1):
  a_t = sigmoid(T.dot(x_t, Wi) + T.dot(a_tm1, Wh) + bh)
  y_t = T.nnet.softmax(T.dot(a_t, Wo) + bo)
  y_t = y_t[0]
  return a_t, y_t

[a_seq, y_seq],_ = theano.scan(
  step,
  sequences = x_seq,
  outputs_info = [a_0, y_0],
  truncate_gradient = -1)

y_seq_last = y_seq[-1][0]

# cost function
cost = T.sum((y_seq - y_hat_seq) ** 2) / 2

# gradient function
gradients = T.grad(cost, parameters)

# training function
rnn_train = theano.function(
    inputs = [x_seq, y_hat_seq],
    outputs = cost,
    #updates = updateFunc(parameters, gradients)
    updates = momentum(parameters, gradients)
)

# testing function
test = theano.function(
    inputs = [x_seq],
    outputs = y_seq_last
)

def run():
  global test_inst
  global mu
  global lamb, movement
  global x_seq, y_hat_seq
  movement = 0
  lamb = 0.5
  mu = 1
  tStart = time.time()
  
  init()
  
  # training information
  f = open("cost.txt", "w+")
  print "start training"
  
  it = 1
  x_seq, y_hat_seq = gen_data()
  while it <= 100:
    cost = rnn_train(x_seq, y_hat_seq)
    print it, " cost: ", cost
    #f.write("%d, cost: %f\n" % (it, cost))
    it += 1
    mu *= 0.9999
    if it % 10 == 0:
      x_seq, y_hat_seq = gen_data()


  tEnd = time.time()
  print "It cost %f mins" % ((tEnd - tStart) / 60)
""" 
  result = test(test_fbank)
  
  f = open("result.csv", "w+")
  f.write("Id,Prediction\n")
  for i in range(len(test_inst)):
    f.write("%s,%s\n" % (test_inst[i], match([result[j][i] for j in range(48)])))
"""
