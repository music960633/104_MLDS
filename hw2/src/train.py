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
N_HIDDEN = 128
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

def gen_data(idx):
  global  map_inst_48
  train_inst, train_fbank = readdata.get_small_train_data(idx)
  X_seq = []
  Y_hat_seq = []
  i = 0
  size =  len(train_fbank[0])
  while i < size:
    X_seq += [[train_fbank[row][i] for row in range(N_INPUT)]]
    Y_hat_seq += [[(1.0 if map_inst_48[train_inst[i]] == map_idx_48[row] else 0.0) for row in range(N_OUTPUT)]]
    i = i + 1
  return X_seq, Y_hat_seq, size


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

def softmax(z):
  return T.exp(z) / T.sum(T.exp(z))

def step (x_t, a_tm1, y_tm1):
  a_t = sigmoid(T.dot(x_t, Wi) + T.dot(a_tm1, Wh) + bh)
  y_t = softmax(T.dot(a_t, Wo) + bo)
  return a_t, y_t

[a_seq, y_seq],_ = theano.scan(
  step,
  sequences = x_seq,
  outputs_info = [a_0, y_0],
  truncate_gradient = -1)

y_seq_last = y_seq[-1][0]

# cost function
cost = T.sum((y_seq - y_hat_seq) ** 2)

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
    outputs = y_seq
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

def validate():
  global map_inst_48, map_inst_48_39
  valid_inst, valid_fbank = readdata.get_small_train_data()
  valid_result = test(valid_fbank)
  data_size = len(valid_inst)
  correct = 0
  for i in range(data_size):
    if map_48_39[map_inst_48[valid_inst[i]]] == match([valid_result[j][i] for j in range(N_INPUT)]):
      correct += 1
  percentage = float(correct) / data_size
  print "validate:", correct, "/", data_size, "(", percentage, ")" 

def run():
  global test_inst
  global mu
  global lamb, movement
  global x_seq, y_hat_seq
  movement = 0
  lamb = 0.5
  mu = 0.01
  tStart = time.time()
  
  init()
  
  # training information
  f = open("cost.txt", "w+")
  print "start training"
  it = 1
  while it <= 1000:
    total_cost = 0
    for i in range(3695):
      x_seq, y_hat_seq , sentence_size = gen_data(i)
      cost = rnn_train(x_seq, y_hat_seq) / float(sentence_size)
      total_cost += cost
      print i, "cost: ", cost
    total_cost /= float(3695)
    print it, "total cost: ", total_cost
    f.write("iteration %s, cost %s" %(i, total_cost))
    a_0.set_value(np.zeros(N_HIDDEN))
    y_0.set_value(np.zeros(N_OUTPUT))
    it += 1
    mu *= 0.9999


  tEnd = time.time()
  print "It cost %f mins" % ((tEnd - tStart) / 60)
  result = test(test_fbank)
  
  f = open("result.csv", "w+")
  f.write("Id,Prediction\n")
  for i in range(len(test_inst)):
    f.write("%s,%s\n" % (test_inst[i], match([result[j][i] for j in range(48)])))
