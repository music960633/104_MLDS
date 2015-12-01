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
batch_size = 256
batch_num = 128

# learning rate
mu = 1.0

# neuron variable declaration
x     = T.matrix("input")
y_hat = T.matrix("reference")
N_INPUT = 69*3
N_HIDDEN = 512
N_OUTPUT = 48
w1    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_INPUT) ] for i in range(N_HIDDEN)]))
w2    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_HIDDEN)] for i in range(N_HIDDEN)]))
w3    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_HIDDEN)] for i in range(N_OUTPUT)]))
b1    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_HIDDEN)]))
b2    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_HIDDEN)]))
b3    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_OUTPUT)]))
w1_mom    = theano.shared(numpy.matrix([[0.0 for j in range(N_INPUT) ] for i in range(N_HIDDEN)]))
w2_mom    = theano.shared(numpy.matrix([[0.0 for j in range(N_HIDDEN)] for i in range(N_HIDDEN)]))
w3_mom    = theano.shared(numpy.matrix([[0.0 for j in range(N_HIDDEN)] for i in range(N_OUTPUT)]))
b1_mom    = theano.shared(numpy.array([0.0 for i in range(N_HIDDEN)]))
b2_mom    = theano.shared(numpy.array([0.0 for i in range(N_HIDDEN)]))
b3_mom    = theano.shared(numpy.array([0.0 for i in range(N_OUTPUT)]))
parameters = [w1, w2, w3, b1, b2, b3]
reg = T.sum(w1*w1) + T.sum(w2*w2) + T.sum(w3*w3)
momentum_params = [w1_mom, w2_mom, w3_mom, b1_mom, b2_mom, b3_mom]

z1 = T.dot(w1, x) + b1.dimshuffle(0, 'x')
a1 = 1.0 / (1 + T.exp(-z1))
z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
a2 = T.switch(z2<0, 0, z2)
z3 = T.dot(w3, a2) + b3.dimshuffle(0, 'x')
y  = T.exp(z3) / T.sum(T.exp(z3), axis=0).dimshuffle('x', 0)

cost = T.sum(-T.log(y) * y_hat) / batch_size + (0.1 * reg)
gradients = T.grad(cost, parameters)
post = T.log(y)

def init():
  print "initializing..."
  global train_inst, train_fbank
  global test_inst, test_fbank
  global map_inst_48, map_48_39
  global map_idx_48, map_48_idx
  # training data
  print "reading training data"
  train_inst, train_fbank = readdata.get_small_train_fbank(0)
  # testing data
  print "reading testing data"
  test_inst , test_fbank  = readdata.get_test_fbank()
  # instance name and phone mapping
  print "reading instance name - phone mapping"
  map_inst_48 = readdata.get_map_inst_48()
  map_48_39   = readdata.get_map_48_39()
  # phone and index mapping
  print "generating phone - index mapping"
  map_48_idx = readdata.get_map_48_idx()
  map_idx_48 = dict(zip(map_48_idx.values(), map_48_idx.keys()))

def change_train_data():
  global train_inst, train_fbank
  print "changing training data"
  idx = int(random.random() * 10)
  train_inst, train_fbank = readdata.get_small_train_fbank(idx)


def make_batch(size, num):
  global train_inst, train_fbank
  global map_inst_48, map_idx_48
  data_size = len(train_inst)
  X_ret = []
  Y_ret = []
  for i in range(num):
    # random select
    idx = [int(random.random() * data_size) for j in range(size)]
    # make batch
    X_batch = [[train_fbank[row][idx[j]] for j in range(size)] for row in range(N_INPUT)]
    Y_batch = [[(1.0 if map_inst_48[train_inst[idx[j]]] == map_idx_48[row] else 0.0) for j in range(size)] for row in range(48)]
    X_ret += [X_batch]
    Y_ret += [Y_batch]
  return X_ret, Y_ret

# update function
def updateFunc(param, grad):
  global mu
  param_updates = [(p, p - mu * g) for p, g in zip(param, grad)]
  return param_updates

def momentum(param, momentum_params, grad):
  global mu
  param_updates = []
  lamb = 0.5
  for p, m, g in zip(param, momentum_params, grad):
    new_m = (lamb * m) - (mu * g)
    param_updates += [(m, new_m)]
    param_updates += [(p, p + new_m)]
  return param_updates

# training function
train = theano.function(
    inputs = [x, y_hat],
    # updates = updateFunc(parameters, gradients),
    updates = momentum(parameters, momentum_params, gradients),
    outputs = cost
)

# testing function
test = theano.function(
    inputs = [x],
    outputs = y
)

get_post = theano.function(
    inputs = [x],
    outputs = post
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
  idx = int(random.random() * 10)
  valid_inst, valid_fbank = readdata.get_small_train_fbank(idx)
  valid_result = test(valid_fbank)
  data_size = len(valid_inst)
  correct = 0
  for i in range(data_size):
    if map_48_39[map_inst_48[valid_inst[i]]] == match([valid_result[j][i] for j in range(48)]):
      correct += 1
  percentage = float(correct) / data_size
  print "validate:", correct, "/", data_size, "(", percentage, ")" 
  return percentage

def run():
  global batch_size, batch_num
  global test_inst
  global mu
  mu = 0.001
  tStart = time.time()
  
  init()
  
  # training information
  print "start training"
  
  it = 1
  mx = 0.0
  while True:
    cost = 0
    X_batch, Y_hat_batch = make_batch(batch_size, batch_num)
    for j in range(batch_num):
      cost += train(X_batch[j], Y_hat_batch[j])
    cost /= batch_num
    print it, " cost: ", cost
    if (it % 20 == 0):
      val = validate()
      f_gen = open("generate.txt", "r")
      s = f_gen.readline().strip()
      f_gen.close()
      if val > mx and s == "yes":
        mx = val
        result = test(test_fbank)
        f = open("result/greenli/new_1.csv", "w+")
        f.write("Id,Prediction\n")
        for i in range(len(test_inst)):
          f.write("%s,%s\n" % (test_inst[i], match([result[j][i] for j in range(48)])))
        f.close()
        
        print "generating my_train.post"
        global train_inst, train_fbank
        f = open("./my_train.post", "w+")
        for idx in range(12):
          train_inst, train_fbank = readdata.get_train_fbank(idx)
          post_result = get_post(train_fbank)
          for i in range(len(train_inst)):
            f.write("%s" % train_inst[i])
            for j in range(48):
              f.write(" %f" % post_result[j][i])
            f.write('\n')
        f.close()

        print "generating my_test.post"
        f = open("./my_test.post", "w+")
        post_result = get_post(test_fbank)
        for i in range(len(test_inst)):
          f.write("%s" % test_inst[i])
          for j in range(48):
            f.write(" %f" % post_result[j][i])
          f.write('\n')
        f.close()
  
        train_inst, train_fbank = readdata.get_small_train_fbank(0)

      # change_train_data()
    it += 1
    mu *= 0.999
