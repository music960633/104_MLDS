import theano
import theano.tensor as T
import numpy
import random
import re
import time

import readdata

# raw data
train_inst  = []
train_fbank = []
test_inst   = []
test_fbank  = []
map_inst_48 = {}
map_48_39   = {}
map_idx_48  = {}
map_48_idx  = {}

# batch size and number
batch_size = 128
batch_num = 1024

# learning rate
mu = 0.1

# neuron variable declaration
x_seq     = T.matrix("input")
y_hat_seq = T.matrix("reference")
Wi   = theano.shared(numpy.matrix([[random.gauss(0.0, 0.1) for j in range(48) ] for i in range(128)]))
Wh   = theano.shared(numpy.matrix([[random.gauss(0.0, 0.1) for j in range(128) ] for i in range(128)]))
Wo   = theano.shared(numpy.matrix([[random.gauss(0.0, 0.1) for j in range(128) ] for i in range(48)]))
bh   = theano.shared(numpy.matrix([[random.gauss(0.0, 0.1)] for i in range(128)]))
bo   = theano.shared(numpy.matrix([[random.gauss(0.0, 0.1)] for i in range(48)]))
a_0  = theano.shared(numpy.matrix([[random.gauss(0.0, 0.1)] for i in range(48)])) 
y_0  = theano.shared(numpy.matrix([[random.gauss(0.0, 0.1)] for i in range(48)]))

parameters = [Wi, Wh, Wo, bh, bo]

def init():
  print "initializing..."
  global train_inst, train_fbank
  global test_inst, test_fbank
  global map_inst_48, map_48_39
  global map_idx_48, map_48_idx
  global map_48_char
  # training data
  print "reading training data"
  train_inst, train_fbank = readdata.get_train_post()
  # testing data
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

momentum = 0.9
lamb = 0.5
# update function
def updateFunc(param, grad):
  global mu, lamb, momentum, movement
  param_updates = []
  for p, g in zip(param, grad):
    movement = theano.shared(0.)
    movement = (lamb * movement) - mu * g
    param_updates += [(p, p + movement)]
  return param_updates
  """
    updates = []
    param_update = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
    updates.append((p, p - mu * param_update))
    updates.append((param_update, momentum * param_update + (1. - momentum) * g))
  return updates
  """

def step (x_t, a_tm1, y_tm1):
  a_t = T.nnet.sigmoid( T.dot(x_t, Wi) + T.dot(a_tm1, Wh) + bh)
  y_t = T.nnet.softmax(T.dot(a_t, Wo) + bo)
  return a_t, y_t

[a_seq, y_seq],_ = theano.scan(
  step,
  sequences = x_seq,
  outputs_info = [a_0, y_0],
  truncate_gradient = -1)

# cost function
cost = T.sum((y_seq - y_hat_seq) ** 2) / batch_size

# gradient function
gradients = T.grad(cost, parameters)

# training function
rnn_train = theano.function(
    inputs = [x_seq, y_hat_seq],
    updates = updateFunc(parameters, gradients),
    outputs = cost
)

# testing function
test = theano.function(
    inputs = [x_seq],
    outputs = y_seq
)

def gen_data():
  global train_fbank, map_inst_48
  X_seq = []
  Y_hat_seq = []
  i = 0
  seq = train_inst[i]
  while True:
    X_seq += [[train_fbank[row][i] for row in range(48)]]
    Y_hat_seq += [[(1.0 if map_inst_48[train_inst[i]] == map_idx_48[row] else 0.0) for row in range(48)]]
    i = i+1
    if i > 2: break
  print X_seq, Y_hat_seq
  return X_seq, Y_hat_seq


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
  while it <= 100:
    #cost = 0
    #X_batch, Y_hat_batch = make_batch(batch_size, batch_num)
    x_seq, y_hat_seq = gen_data()
    cost = rnn_train(x_seq, y_hat_seq)
    print it, " cost: ", cost
    f.write("%d, cost: %f\n" % (it, cost))
    it += 1
    mu *= 0.9999

  tEnd = time.time()
  f.write("It cost %f mins" % ((tEnd - tStart) / 60))
  f.close()
  
  result = test(test_fbank)
  
  f = open("result.csv", "w+")
  f.write("Id,Prediction\n")
  for i in range(len(test_inst)):
    f.write("%s,%s\n" % (test_inst[i], match([result[j][i] for j in range(48)])))
