import theano
import theano.tensor as T
import numpy as np
import random
import re
import time


import readdata

# raw data
test_inst   = []
test_post   = []
map_inst_48 = {}
map_48_39   = {}
map_idx_48  = {}
map_48_idx  = {}

# learning rate
mu = 0.001

# parameter
N_HIDDEN = 64
N_INPUT = 48
N_OUTPUT = 48

# neuron variable declaration
x_seq     = T.matrix("input")
y_hat_seq = T.matrix("reference")
Wi   = theano.shared(np.matrix([[random.gauss(0.0, 0.01) for j in range(N_HIDDEN)] for i in range(N_INPUT )]))
Wh   = theano.shared(np.matrix([[0.01 if i==j else 0.00  for j in range(N_HIDDEN)] for i in range(N_HIDDEN)]))
Wo   = theano.shared(np.matrix([[random.gauss(0.0, 0.01) for j in range(N_OUTPUT)] for i in range(N_HIDDEN)]))
bo   = theano.shared(np.array ([ random.gauss(0.0, 0.01) for i in range(N_OUTPUT)]))
bh   = theano.shared(np.array ([ random.gauss(0.0, 0.01) for i in range(N_HIDDEN)]))
a_0 = theano.shared(np.zeros(N_HIDDEN))
y_0 = theano.shared(np.zeros(N_OUTPUT))
parameters = [Wi, bh, Wo, bo, Wh]


def init():
  print "initializing..."
  global train_inst, train_post
  global test_inst, test_post
  global map_inst_48, map_48_39
  global map_idx_48, map_48_idx
  global map_48_char
  print "reading testing data"
  test_inst , test_post  = readdata.get_test_post()
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
  train_inst, train_post = readdata.get_small_train_data(idx)
  X_seq = train_post
  Y_hat_seq = []
  size = len(train_post)
  i = 0
  while i < size:
    Y_hat_seq += [[(1.0 if map_inst_48[train_inst[i]] == map_idx_48[row] else 0.0) for row in range(N_OUTPUT)]]
    i = i + 1
  return X_seq, Y_hat_seq, size


# update function
def updateFunc(param, grad):
  global mu
  parameters_updates = [(p, p - mu * g) for p,g in zip(parameters,gradients) ] 
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
  # y_t = sigmoid(T.dot(a_t, Wo) + bo)
  return a_t, y_t

[a_seq, y_seq], _ = theano.scan(
  step,
  sequences = x_seq,
  outputs_info = [a_0, y_0],
  truncate_gradient = -1
)

# cost function
cost = T.sum(-T.log(y_seq) * y_hat_seq)
# cost = T.sum((y_seq - y_hat_seq) ** 2)

# gradient function
gradients = T.grad(cost, parameters)

# training function
rnn_train = theano.function(
    inputs = [x_seq, y_hat_seq],
    outputs = [cost, y_seq],
    updates = updateFunc(parameters, gradients)
    # updates = momentum(parameters, gradients)
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

def argmax(arr):
  mx = arr[0]
  idx = 0
  i = 0
  for x in arr:
    if x > mx:
      mx = x
      idx = i
    i = i + 1
  return idx, mx

def accuracy(y_seq, y_hat_seq):
  cnt = 0
  for i in range(len(y_seq)):
    if y_hat_seq[i][argmax(y_seq[i])[0]] == 1:
      cnt = cnt + 1
  return float(cnt) / len(y_seq)


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
  print "start training"
  it = 1
  while True:
    num_file = 1
    total_cost = 0
    total_acc  = 0
    max_acc = 0
    for i in range(num_file):
      a_0.set_value(np.zeros(N_HIDDEN))
      y_0.set_value(np.zeros(N_OUTPUT))
      x_seq, y_hat_seq , sentence_size = gen_data(i)
      cost, y_seq = rnn_train(x_seq, y_hat_seq)
      cost = cost / float(sentence_size)
      acc = accuracy(y_seq, y_hat_seq)
      total_cost += cost
      total_acc += acc
      # print i, "cost:", cost, "accuracy:", acc
    
    total_cost /= float(num_file)
    total_acc  /= float(num_file)
    print it, "total cost:", total_cost, "total accuracy:", total_acc
    f = open("../result/cost.txt", "a+")
    f.write("iteration %d, cost %f, accuracy %f\n" % (it, total_cost, total_acc))
    f.close()
    it += 1
    mu *= 0.9999

    if it % 100 == 0 and total_acc > max_acc:
      max_acc = total_acc
      result = test(test_post)
      f = open("../result/result" + str(it/100) + ".csv", "w+")
      f.write("Id,Prediction\n")
      for i in range(len(test_inst)):
        f.write("%s,%s\n" % (test_inst[i], match(result[i])))
      f.close()
