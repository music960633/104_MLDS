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
batch_num = 512

# learning rate
mu = 0.001
decay = 0.9999

# momentum
momentum = 0.0

# neuron variable declaration
x     = T.matrix("input")
y_hat = T.matrix("reference")
w1    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.1) for j in range(69) ] for i in range(514)]))
w2    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.1) for j in range(514)] for i in range(514)]))
w3    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.1) for j in range(514)] for i in range(48) ]))
b1    = theano.shared(numpy.array([random.gauss(0.0, 0.1) for i in range(514)]))
b2    = theano.shared(numpy.array([random.gauss(0.0, 0.1) for i in range(514)]))
b3    = theano.shared(numpy.array([random.gauss(0.0, 0.1) for i in range(48) ]))
parameters = [w1, w2, w3, b1, b2, b3]

z1 = T.dot(w1,  x) + b1.dimshuffle(0, 'x')
a1 = 1.0 / (1 + T.exp(-z1))
z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
a2 = 1.0 / (1 + T.exp(-z2))
z3 = T.dot(w3, a2) + b3.dimshuffle(0, 'x')
mean = T.sum(T.exp(z3), axis=0)
y  = T.exp(z3) / mean.dimshuffle('x', 0)

# cost function
cost = T.sum((y - y_hat) ** 2) / batch_size

# gradient function
gradients = T.grad(cost, parameters)
prev_grad = None

def init():
  print "initializing..."
  global train_inst, train_fbank
  global test_inst, test_fbank
  global map_inst_48, map_48_39
  global map_idx_48, map_48_idx
  # training data
  print "reading training data"
  train_inst, train_fbank = readdata.get_small_train_fbank()
  # testing data
  print "reading testing data"
  test_inst , test_fbank  = readdata.get_test_fbank()
  # instance name and phone mapping
  print "reading instance name - phone mapping"
  map_inst_48 = readdata.get_map_inst_48()
  map_48_39   = readdata.get_map_48_39()
  # phone and index mapping
  print "generating phone - index mapping"
  map_idx_48  = dict(enumerate(map_48_39.keys(), 0))
  map_48_idx  = dict(zip(map_idx_48.values(), map_idx_48.keys()))

def change_train_data():
  global train_inst, train_fbank
  print "changing training data"
  train_inst, train_fbank = readdata.get_small_train_fbank()


def make_batch(size, num):
  global train_inst, train_fbank
  global map_inst_48, map_idx_48
  data_size = len(train_inst)
  X_ret = []
  Y_ret = []
  for i in range(num):
    # random select
    idx = [random.randint(0, data_size-1) for j in range(size)]
    # make batch
    X_batch = [[train_fbank[row][idx[j]] for j in range(size)] for row in range(69)]
    Y_batch = [[(1.0 if map_inst_48[train_inst[idx[j]]] == map_idx_48[row] else 0.0) for j in range(size)] for row in range(48)]
    X_ret += [X_batch]
    Y_ret += [Y_batch]
  return X_ret, Y_ret

# update function
def updateFunc(param, grad):
  global mu, momentum, prev_grad
  # prev_grad = [momentum * g1 + mu * g2 for g1, g2 in zip(prev_grad, grad)] if prev_grad != None \
  #      else [mu * g2 for g2 in grad]
  param_updates = [(p, p - g) for p, g in zip(param, grad)]
  return param_updates

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

def validate():
  global map_inst_48, map_inst_48_39
  valid_inst, valid_fbank = readdata.get_small_train_fbank()
  valid_result = test(valid_fbank)
  data_size = len(valid_inst)
  correct = 0
  for i in range(data_size):
    if map_48_39[map_inst_48[valid_inst[i]]] == match([valid_result[j][i] for j in range(48)]):
      correct += 1
  percentage = float(correct) / data_size
  print "validate:", correct, "/", data_size, "(", percentage, ")" 
  return percentage > 0.65

def run():
  global batch_size, batch_num
  global test_inst
  global mu, decay, momentum
  tStart = time.time()
  
  init()
  
  # training information
  f = open("cost.txt", "w+")
  f.write("train data: small + change data\n")
  f.write("mu = %f, decay = %f\n" % (mu, decay))
  f.write("activation function: ReLU\n")
  f.write("batch_size = %d\n" % batch_size)
  f.write("batch_num = %d\n" % batch_num)
  f.write("3 layers: 69-256-256-48\n")
  
  print "start training"
  
  it = 1
  while it <= 1000:
    cost = 0
    X_batch, Y_hat_batch = make_batch(batch_size, batch_num)
    for j in range(batch_num):
      cost += train(X_batch[j], Y_hat_batch[j])
    cost /= batch_num
    print it, " cost: ", cost
    f.write("%d, cost: %f\n" % (it, cost))
    if (it % 20 == 0):
       if validate():
         break
       change_train_data()
    it += 1
    mu *= decay

  tEnd = time.time()
  f.write("It cost %f mins" % ((tEnd - tStart) / 60))
  f.close()
  
  result = test(test_fbank)
  
  f = open("result/result_music960633_5.csv", "w+")
  f.write("Id,Prediction\n")
  for i in range(len(test_inst)):
    f.write("%s,%s\n" % (test_inst[i], match([result[j][i] for j in range(48)])))
  #for i in range(len(train_inst)):
    #f.write("%s,%s\n" % (train_inst[i], match([result[j][i] for j in range(48)])))
  f.close()