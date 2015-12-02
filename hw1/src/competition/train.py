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
batch_num = 128

# learning rate
mu = 1.0

# neuron variable declaration
x     = T.matrix("input")
y_hat = T.matrix("reference")
N_INPUT = 69*3
N_HIDDEN = 500
N_OUTPUT = 48
w1    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_INPUT) ] for i in range(N_HIDDEN)]))
w2    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_HIDDEN)] for i in range(N_HIDDEN)]))
w3    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_HIDDEN)] for i in range(N_HIDDEN)]))
w4    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_HIDDEN)] for i in range(N_OUTPUT)]))
b1    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_HIDDEN)]))
b2    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_HIDDEN)]))
b3    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_HIDDEN)]))
b4    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_OUTPUT)]))
w1_ada    = theano.shared(numpy.matrix([[1.0 for j in range(N_INPUT) ] for i in range(N_HIDDEN)]))
w2_ada    = theano.shared(numpy.matrix([[1.0 for j in range(N_HIDDEN)] for i in range(N_HIDDEN)]))
w3_ada    = theano.shared(numpy.matrix([[1.0 for j in range(N_HIDDEN)] for i in range(N_HIDDEN)]))
w4_ada    = theano.shared(numpy.matrix([[1.0 for j in range(N_HIDDEN)] for i in range(N_OUTPUT)]))
b1_ada    = theano.shared(numpy.array([1.0 for i in range(N_HIDDEN)]))
b2_ada    = theano.shared(numpy.array([1.0 for i in range(N_HIDDEN)]))
b3_ada    = theano.shared(numpy.array([1.0 for i in range(N_HIDDEN)]))
b4_ada    = theano.shared(numpy.array([1.0 for i in range(N_OUTPUT)]))
parameters = [w1, w2, w3, w4, b1, b2, b3, b4]
adagrad_params = [w1_ada, w2_ada, w3_ada, w4_ada, b1_ada, b2_ada, b3_ada, b4_ada]

z1 = T.dot(w1, x ) + b1.dimshuffle(0, 'x')
a1 = 1.0 / (1 + T.exp(-z1))
z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
a2 = 1.0 / (1 + T.exp(-z2))
z3 = T.dot(w3, a2) + b3.dimshuffle(0, 'x')
a3 = 1.0 / (1 + T.exp(-z3))
z4 = T.dot(w4, a3) + b4.dimshuffle(0, 'x')
y  = T.exp(z4) / T.sum(T.exp(z4), axis=0).dimshuffle('x', 0)

cost = T.sum(-T.log(y) * y_hat) / batch_size
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
def adagrad(param, adagrad_params, grad):
  global mu
  param_updates = []
  for p, a, g in zip(param, adagrad_params, grad):
    param_updates += [(p, p - mu * (g / a))]
    param_updates += [(a, a + g*g)]
  return param_updates

# training function
train = theano.function(
    inputs = [x, y_hat],
    updates = adagrad(parameters, adagrad_params, gradients),
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
  idx = int(random.random() * 5) + 1
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
    if (it % 50 == 0):
      val = validate()
      f_gen = open("generate.txt", "r")
      s = f_gen.readline().strip()
      f_gen.close()
      if val > mx:
        mx = val
        result = test(test_fbank)
        f = open("result/new.csv", "w+")
        f.write("Id,Prediction\n")
        for i in range(len(test_inst)):
          f.write("%s,%s\n" % (test_inst[i], match([result[j][i] for j in range(48)])))
        f.close()
        
      if s == "yes":
        print "generating my_train.post"
        f = open("./my_train.post", "w+")
        for idx in range(12):
          train_inst_all, train_fbank_all = readdata.get_train_fbank(idx)
          post_result = get_post(train_fbank_all)
          for i in range(len(train_inst_all)):
            f.write("%s" % train_inst_all[i])
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
  
    it += 1
    mu *= 0.999
