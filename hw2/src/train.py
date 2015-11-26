import theano
import theano.tensor as T
import numpy as np
import random
import re
import time


import readdata

# raw data
map_inst_48 = {}
map_48_39   = {}
map_idx_48  = {}
map_48_idx  = {}
map_48_char = {}

# learning rate
mu = 0.0005

# parameter
N_HIDDEN = 64
N_INPUT = 48
N_OUTPUT = 48

# neuron variable declaration
x_seq     = T.matrix("input")
y_hat_seq = T.matrix("reference")
Wi   = theano.shared(np.matrix([[random.gauss(0.0, 0.001) for j in range(N_HIDDEN)] for i in range(N_INPUT )]))
Wh   = theano.shared(np.matrix([[0.01 if i==j else 0.00  for j in range(N_HIDDEN)] for i in range(N_HIDDEN)]))
Wo   = theano.shared(np.matrix([[random.gauss(0.0, 0.001) for j in range(N_OUTPUT)] for i in range(N_HIDDEN)]))
bo   = theano.shared(np.array ([ random.gauss(0.0, 0.001) for i in range(N_OUTPUT)]))
bh   = theano.shared(np.array ([ random.gauss(0.0, 0.001) for i in range(N_HIDDEN)]))

Wi_sigma   = theano.shared(np.matrix([[0.0 for j in range(N_HIDDEN)] for i in range(N_INPUT )]))
Wh_sigma   = theano.shared(np.matrix([[0.0 for j in range(N_HIDDEN)] for i in range(N_HIDDEN)]))
Wo_sigma   = theano.shared(np.matrix([[0.0 for j in range(N_OUTPUT)] for i in range(N_HIDDEN)]))
b1_sigma   = theano.shared(np.array ([ 0.0 for i in range(N_HIDDEN)]))
b2_sigma   = theano.shared(np.array ([ 0.0 for i in range(N_HIDDEN)]))
bo_sigma   = theano.shared(np.array ([ 0.0 for i in range(N_OUTPUT)]))
a_0 = theano.shared(np.zeros(N_HIDDEN))
y_0 = theano.shared(np.zeros(N_OUTPUT))
parameters = [W1, b1, W2, Wm, b2, Wo, bo]
sigma = [W1_sigma, b1_sigma, W2_sigma, Wm_sigma, b2_sigma, Wo_sigma, bo_sigma]


def init():
  print "initializing..."
  global train_inst, train_post
  global test_inst, test_post
  global map_inst_48, map_48_39
  global map_idx_48, map_48_idx
  global map_48_char
  # instance name and phone mapping
  print "reading instance name - phone mapping"
  map_inst_48 = readdata.get_map_inst_48()
  map_48_39   = readdata.get_map_48_39()
  map_48_char = readdata.get_map_48_char()
  # phone and index mapping
  print "generating phone - index mapping"
  map_idx_48  = dict(enumerate(map_48_39.keys(), 0))
  map_48_idx  = dict(zip(map_idx_48.values(), map_idx_48.keys()))

def get_data(idx):
  global map_inst_48
  train_inst, train_post = readdata.get_small_train_data(idx)
  X_seq = train_post
  Y_hat_seq = []
  size = len(train_post)
  for i in range(size):
    Y_hat_seq += [[(1.0 if map_inst_48[train_inst[i]] == map_idx_48[row] else 0.0) for row in range(N_OUTPUT)]]
  return X_seq, Y_hat_seq, size


# update function
def updateFunc(param, grad):
  global mu
  parameters_updates = [(p, p - mu * T.clip(g, -0.01, 0.01)) for p,g in zip(parameters,gradients) ] 
  return parameters_updates

def rmsprop (param, sigma, grad):
  global mu
  alpha = 0.7
  param_updates = []
  for p, s, g in zip(param, sigma, grad):
    g = T.clip(g, -1.0, 1.0)
    new_s = T.sqrt(alpha * T.sqr(s) + (1 - alpha) * T.sqr(g))
    param_updates += [(p, p - mu * g / new_s)]
    param_updates += [(s, new_s)]
  return param_updates

def sigmoid(z):
  return 1/(1 + T.exp(-z))

def relu(z):
  return T.log(1 + T.exp(z))

def softmax(zs):
  return T.exp(zs) / T.sum(T.exp(zs), axis=1).dimshuffle(0, 'x')

def step (a1_t, a_tm1):
  return sigmoid(T.dot(a1_t, W2) + T.dot(a_tm1, Wm) + b2)

a1_seq = sigmoid(T.dot(x_seq, W1) + b1.dimshuffle('x', 0))
a2_seq, _ = theano.scan(
  step,
  sequences = a1_seq,
  outputs_info = a_0,
  truncate_gradient = -1
)
y_seq = softmax(T.dot(a2_seq, Wo) + bo.dimshuffle('x', 0))

# cost function
cost = T.sum(-T.log(y_seq) * y_hat_seq)

# gradient function
gradients = T.grad(cost, parameters)

# training function
rnn_train = theano.function(
    inputs = [x_seq, y_hat_seq],
    outputs = [cost, y_seq],
    #updates = updateFunc(parameters, gradients)
    updates = rmsprop(parameters, sigma, gradients)
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

def trim(s):
  tmp = ""
  ret = ""
  window_size = 5
  for i in range(len(s) - window_size):
    count = {}
    for j in range(window_size):
      if s[i+j] not in count.keys():
        count[s[i+j]] = 1
      else:
        count[s[i+j]] += 1
    for phen, num in count.items():
      if num > 2:
        tmp += phen
        break

  for i in range(len(tmp)):
    if i == 0 or tmp[i] != tmp[i-1]:
      ret += tmp[i]
  return ret

def gen_test(idx):
  global map_48_char
  print "generating result..."
  f1 = open("../result/result_" + str(idx) + ".csv", "w+")
  f2 = open("../result/phone_" + str(idx) + ".csv", "w+")
  f1.write("id,phone_sequence\n")
  f2.write("id,phone")
  for i in range(592):
    a_0.set_value(np.zeros(N_HIDDEN))
    y_0.set_value(np.zeros(N_OUTPUT))
    test_inst, x_seq = readdata.get_small_test_data(i)
    result = test(x_seq)
    seq = ""
    f2.write("%s" % test_inst)
    for j in range(len(result)):
      ch = map_48_char[match(result[j])]
      if ch != 'L':
        seq += ch
      f2.write(",%s" % match(result[j]))
    f1.write("%s,%s\n" % (test_inst, trim(seq)))
    f2.write("\n")
  f1.close()
  f2.close()

def run():
  global test_inst
  global mu
  global x_seq, y_hat_seq
  tStart = time.time()
 
  init()

  # training information
  print "start training"
  f = open("../result/cost.csv", "a+")
  f.write("iteration,cost,accuracy\n")
  f.close()
  it = 1
  num_file = 13000
  gap = 1000
  max_acc = 0.0
  watermark = 1
<<<<<<< HEAD
  threshold = 0.6
  mu = 0.0001
=======
  # num_file = 13064
>>>>>>> 5df003fc6df85f4e0ee0f840faa8541a8f8b13b6
  while True:
    total_cost = 0
    total_acc  = 0
    for i in range(num_file):
      a_0.set_value(np.zeros(N_HIDDEN))
      y_0.set_value(np.zeros(N_OUTPUT))
      x_seq, y_hat_seq, sentence_size = get_data(i)
      cost, y_seq = rnn_train(x_seq, y_hat_seq)
      cost = cost / float(sentence_size)
      acc = accuracy(y_seq, y_hat_seq)
      total_cost += cost
      total_acc += acc
    
    total_cost /= float(num_file)
    total_acc  /= float(num_file)
    print it, "num_file:", num_file, "total cost:", total_cost, "total accuracy:", total_acc
    f = open("../result/cost.csv", "a+")
    f.write("%d,%f,%f\n" % (it, total_cost, total_acc))
    f.close()
    it += 1
<<<<<<< HEAD
    mu *= 0.999
    if total_acc > threshold:
=======
    mu *= 0.9999
    if total_acc > 0.9:
>>>>>>> 5df003fc6df85f4e0ee0f840faa8541a8f8b13b6
      gen_test(watermark)
      watermark += 1
      threshold *= 1.05
