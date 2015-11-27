import math

"""
phone id
0~47 : normal
48   : seperate (start and end)
"""
N_PHONE = 48 + 1

map_phone_to_idx = {}
map_inst_to_phone = {}
p_emit = [[0.0 for j in range(N_PHONE)] for i in range(N_PHONE)]
p_tran = [[0.0 for j in range(N_PHONE)] for i in range(N_PHONE)]

def build():
  # build phone -> idx mapping
  print "reading 48_39.map"
  global map_phone_to_idx
  f_phone = open("../../data/MLDS_HW1_RELEASE_v1/phones/48_39.map", "r")
  idx = 0
  for line in f_phone:
    tokens = line.strip().split('\t')
    assert len(tokens) == 2, "parse phone error"
    assert tokens[0] not in map_phone_to_idx, "phone already exists"
    map_phone_to_idx[tokens[0]] = idx
    idx += 1
  
  # build inst -> phone mapping and phone transition probability
  print "reading train.lab"
  global map_inst_to_phone
  global p_tran
  f_label = open("../../data/MLDS_HW1_RELEASE_v1/label/train.lab", "r")
  prev_phone_idx = 0
  for line in f_label:
    tokens = line.strip().split(',')
    assert len(tokens) == 2, "parse label error"
    assert tokens[0] not in map_inst_to_phone, "inst already exists"
    inst = tokens[0]
    phone = tokens[1]
    phone_idx = map_phone_to_idx[phone]
    map_inst_to_phone[inst] = phone
    if inst.split('_')[2] != "1":
      p_tran[prev_phone_idx][phone_idx] += 1
    else:
      if prev_phone_idx != -1:
        p_tran[prev_phone_idx][48] += 1
      p_tran[48][phone_idx] += 1
    prev_phone_idx = phone_idx

  # build phone -> prediction probability
  print "reading posteriorgram"
  global p_emit
  f_data = open("my_train.post", "r")
  for line in f_data:
    tokens = line.strip().split(' ')
    assert len(tokens) == 49, "parse fbank error"
    inst = tokens[0]
    prob = map(lambda x: math.exp(float(x)), tokens[1:])
    phone = map_inst_to_phone[inst]
    phone_idx = map_phone_to_idx[phone]
    prediction = max(zip(prob, range(48)))[1]
    p_emit[phone_idx][prediction] += 1


  # divide by sum, occurance to probability
  for l in p_tran:
    s = sum(l)
    if s == 0: s = 1.0
    l = map(lambda x: float(x)/s, l)
  for l in p_emit:
    s = sum(l)
    if s == 0: s = 1.0
    l = map(lambda x: float(x)/s, l)

def viterbi(predictions):
  len_pred = len(predictions)
  dp = [[0.0 for j in range(48)] for i in range(len_pred)]
  prev = [[0.0 for j in range(48)] for i in range(len_pred)]

  # start
  for j in range(48):
    dp[0][j] = p_tran[48][j] * p_emit[j][predictions[i]]
    prev[0][j] = -1

  # sequence
  for i in range(1, len_pred):
    for j in range(48):
      for k in range(48):
        prob = dp[i-1][k] * p_tran[k][j] * p_emit[j][predictions[i]]
        if prob > dp[i][j]:
          dp[i][j] = prob
          prev[i][j] = k
  # end
  end_prob = 0.0
  idx = 0
  for k in range(48):
    prob = dp[len_pred-1][k] * p_tran[k][48]
    if prob > end_prob:
      end_prob = prob
      idx = k

  # backtrack
  seq = []
  for i in range(len_pred-1, -1, -1):
    seq = [idx] + seq
    idx = prev[i][idx]
    
  return seq

build()
print viterbi([10]*100)
