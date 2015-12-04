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

# validate data size
N_VALID_SIZE = 10000

# learning rate
mu = 0.001
lamda = 0.001

# neuron variable declaration
x     = T.matrix("input")
y_hat = T.matrix("reference")
N_EXT = 2
N_INPUT = 69
N_EXTINPUT = N_INPUT * (2*N_EXT + 1)
N_HIDDEN = 256
N_OUTPUT = 48
w1    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_EXTINPUT)] for i in range(N_HIDDEN)]))
w2    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_HIDDEN)  ] for i in range(N_HIDDEN)]))
w3    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_HIDDEN)  ] for i in range(N_HIDDEN)]))
w4    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_HIDDEN)  ] for i in range(N_HIDDEN)]))
w5    = theano.shared(numpy.matrix([[random.gauss(0.0, 0.01) for j in range(N_HIDDEN)] for i in range(N_OUTPUT)]))
b1    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_HIDDEN)]))
b2    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_HIDDEN)]))
b3    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_HIDDEN)]))
b4    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_HIDDEN)]))
b5    = theano.shared(numpy.array([random.gauss(0.0, 0.01) for i in range(N_OUTPUT)]))
w1_ada    = theano.shared(numpy.matrix([[1.0 for j in range(N_EXTINPUT)] for i in range(N_HIDDEN)]))
w2_ada    = theano.shared(numpy.matrix([[1.0 for j in range(N_HIDDEN)  ] for i in range(N_HIDDEN)]))
w3_ada    = theano.shared(numpy.matrix([[1.0 for j in range(N_HIDDEN)  ] for i in range(N_HIDDEN)]))
w4_ada    = theano.shared(numpy.matrix([[1.0 for j in range(N_HIDDEN)  ] for i in range(N_HIDDEN)]))
w5_ada    = theano.shared(numpy.matrix([[1.0 for j in range(N_HIDDEN)  ] for i in range(N_OUTPUT)]))
b1_ada    = theano.shared(numpy.array([1.0 for i in range(N_HIDDEN)]))
b2_ada    = theano.shared(numpy.array([1.0 for i in range(N_HIDDEN)]))
b3_ada    = theano.shared(numpy.array([1.0 for i in range(N_HIDDEN)]))
b4_ada    = theano.shared(numpy.array([1.0 for i in range(N_HIDDEN)]))
b5_ada    = theano.shared(numpy.array([1.0 for i in range(N_OUTPUT)]))
parameters = [w1, w2, w3, w4, w5, b1, b2, b3, b4, b5]
adagrad_params = [w1_ada, w2_ada, w3_ada, w4_ada, w5_ada, b1_ada, b2_ada, b3_ada, b4_ada, b5_ada]
reg_param = T.sum(w1*w1) + T.sum(w2*w2) + T.sum(w3*w3) + T.sum(w4*w4) + T.sum(w5*w5)

z1 = T.dot(w1, x ) + b1.dimshuffle(0, 'x')
a1 = 1.0 / (1 + T.exp(-z1))
z2 = T.dot(w2, a1) + b2.dimshuffle(0, 'x')
a2 = T.switch(z2<0, 0.1*z2, z2)
z3 = T.dot(w3, a2) + b3.dimshuffle(0, 'x')
a3 = T.switch(z3<0, 0.1*z3, z3)
z4 = T.dot(w4, a3) + b4.dimshuffle(0, 'x')
a4 = T.switch(z4<0, 0.1*z4, z4)
z5 = T.dot(w5, a4) + b5.dimshuffle(0, 'x')
y  = T.exp(z5) / T.sum(T.exp(z5), axis=0).dimshuffle('x', 0)

cost = (-T.sum(y_hat * T.log(y) + (1-y_hat) * T.log(1-y)) + 0.5 * lamda * reg_param)/ batch_size
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
  train_inst, train_fbank = readdata.get_train_fbank()
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


def clip(val, mn, mx):
  assert mn <= mx, "mn is larger than mx"
  return min(max(val, mn), mx)

def make_batch(size, num):
  data_size = len(train_inst)
  X_ret = []
  Y_ret = []
  for i in range(num):
    # random select
    idx = [int(random.random() * data_size) for j in range(size)]
    # make batch
    X_batch = [ [train_fbank[r][clip(idx[j]+e, 0, data_size-1)] for j in range(size)] \
        for e in range(-N_EXT, N_EXT+1) for r in range(N_INPUT) ]
    Y_batch = [ [(1.0 if map_inst_48[train_inst[idx[j]]] == map_idx_48[r] else 0.0) for j in range(size)] \
        for r in range(N_OUTPUT) ]
    X_ret += [X_batch]
    Y_ret += [Y_batch]
  return X_ret, Y_ret

def make_full_batch(fbank):
  data_size = len(fbank[0])
  # make batch
  X_batch = [ [fbank[r][clip(j+e, 0, data_size-1)] for j in range(data_size)] \
      for e in range(-N_EXT, N_EXT+1) for r in range(N_INPUT)  ]
  return X_batch


# update function
def adagrad(param, adagrad_params, grad):
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

def argmax(arr):
  idx = 0
  for i in range(len(arr)):
    if arr[i] > arr[idx]:
      idx = i
  return idx

def validate():
  valid_x, valid_yhat = make_batch(N_VALID_SIZE, 1)
  valid_x = valid_x[0]
  valid_yhat = valid_yhat[0]
  valid_y = test(valid_x)
  correct = 0
  for i in range(N_VALID_SIZE):
    if argmax([valid_yhat[j][i] for j in range(N_OUTPUT)]) == \
        argmax([valid_y[j][i] for j in range(N_OUTPUT)]):
      correct += 1
  percentage = float(correct) / N_VALID_SIZE
  print "validate:", correct, "/", N_VALID_SIZE, "(", percentage, ")" 
  return percentage

def cost_init():
  f = open("cost.csv", "w+")
  f.write("iteration,cost\n")
  f.close()

def cost_report(it, cst):
  print it, "cost:", cst
  f = open("cost.csv", "a+")
  f.write("%d,%f\n" % (it, cst))
  f.close()


def run():
  init()
  cost_init()
  
  # training information
  print "start training"

  it = 0
  mx = 0.0
  while True:
    it += 1
    cst = 0
    X_batch, Y_hat_batch = make_batch(batch_size, batch_num)
    for j in range(batch_num):
      cst += train(X_batch[j], Y_hat_batch[j]) / batch_num
    cost_report(it, cst)
    
    if (it % 5 == 0):
      val = validate()
      f_gen = open("generate.txt", "r")
      s = f_gen.readline().strip()
      f_gen.close()
      if val > mx:
        mx = val
        result = test(make_full_batch(test_fbank))
        f = open("result/out.csv", "w+")
        f.write("Id,Prediction\n")
        size = len(test_inst)
        for i in range(size):
          idx = argmax([result[j][i] for j in range(N_OUTPUT)])
          f.write("%s,%s\n" % (test_inst[i], map_48_39[map_idx_48[idx]]))
        f.close()
        
      if s == "yes":
        print "generating my_train.post"
        f = open("./my_train.post", "w+")
        for idx in range(12):
          train_inst_all, train_fbank_all = readdata.get_seg_train_fbank(idx)
          post_result = get_post(make_full_batch(train_fbank_all))
          size = len(train_inst_all)
          for i in range(size):
            f.write("%s" % train_inst_all[i])
            for j in range(N_OUTPUT):
              f.write(" %f" % post_result[j][i])
            f.write('\n')
        f.close()

        print "generating my_test.post"
        f = open("./my_test.post", "w+")
        post_result = get_post(make_full_batch(test_fbank))
        size = len(test_inst)
        for i in range(size):
          f.write("%s" % test_inst[i])
          for j in range(N_OUTPUT):
            f.write(" %f" % post_result[j][i])
          f.write('\n')
        f.close()
  
