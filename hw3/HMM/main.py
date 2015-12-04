import math

"""
phone id
0~47 : normal
48   : seperate (start and end)
"""
N_PHONE = 48 + 1
INF = 1000000.0

map_phone_to_idx = {}
map_idx_to_phone = {}
map_48_to_39 = {}
map_inst_to_phone = {}
map_phone_to_char = {}
p_emit_1 = [[0.000001 for j in range(N_PHONE)] for i in range(N_PHONE)]
p_emit_2 = [[0.000001 for j in range(N_PHONE)] for i in range(N_PHONE)]
p_tran = [[0.000001 for j in range(N_PHONE)] for i in range(N_PHONE)]

def build():
  # build phone -> idx mapping
  print "reading 48_39.map"
  global map_phone_to_idx, map_idx_to_phone, map_48_to_39
  f_phone = open("../../data/MLDS_HW1_RELEASE_v1/phones/48_39.map", "r")
  idx = 0
  for line in f_phone:
    tokens = line.strip().split('\t')
    assert len(tokens) == 2, "parse phone error"
    assert tokens[0] not in map_phone_to_idx, "phone already exists"
    map_phone_to_idx[tokens[0]] = idx
    map_idx_to_phone[idx] = tokens[0]
    map_48_to_39[tokens[0]] = tokens[1]
    idx += 1

  print "reading 48_idx_chr.map_b"
  global map_phone_to_char
  f_chr = open("../../data/48_idx_chr.map_b")
  for line in f_chr:
    tokens = line.strip().replace('\t', ' ').split(' ')
    tokens = [token for token in tokens if token != '']
    assert len(tokens) == 3, "parse chr error"
    assert tokens[0] not in map_phone_to_char, "phone already exists"
    map_phone_to_char[tokens[0]] = tokens[2]
  
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
  global p_emit_1, p_emit_2
  f_data = open("my_train.post", "r")
  for line in f_data:
    tokens = line.strip().split(' ')
    assert len(tokens) == 49, "parse fbank error"
    inst = tokens[0]
    prob = map(lambda x: math.exp(float(x)), tokens[1:])
    phone = map_inst_to_phone[inst]
    phone_idx = map_phone_to_idx[phone]
    prediction = max(zip(prob, range(48)))[1]
    # use argmax
    p_emit_1[phone_idx][prediction] += 1
    # use phone probability
    for i in range(len(prob)):
      p_emit_2[phone_idx][i] += prob[i]


  # divide by sum, occurance to probability
  for i in range(len(p_tran)):
    s = sum(p_tran[i])
    p_tran[i] = map(lambda x: float(x)/s, p_tran[i])
  for i in range(len(p_emit_1)):
    s = sum(p_emit_1[i])
    p_emit_1[i] = map(lambda x: float(x)/s, p_emit_1[i])
  for i in range(len(p_emit_2)):
    s = sum(p_emit_2[i])
    p_emit_2[i] = map(lambda x: float(x)/s, p_emit_2[i])

  print p_tran[36]
  print p_emit_1[36]

def viterbi(predictions, tran, emit):
  len_pred = len(predictions)
  dp = [[-INF for j in range(48)] for i in range(len_pred)]
  prev = [[0 for j in range(48)] for i in range(len_pred)]

  # start
  for j in range(48):
    # dp[0][j] = math.log(tran[48][j]) + math.log(emit[j][predictions[0]])
    dp[0][j] = math.log(emit[j][predictions[0]])
    prev[0][j] = -1

  # sequence
  for i in range(1, len_pred):
    for j in range(48):
      for k in range(48):
        prob = dp[i-1][k] + math.log(tran[k][j]) + math.log(emit[j][predictions[i]])
        if prob > dp[i][j]:
          dp[i][j] = prob
          prev[i][j] = k
  # end
  end_prob = -INF
  idx = 0
  for k in range(48):
    # prob = dp[len_pred-1][k] + math.log(tran[k][48])
    prob = dp[len_pred-1][k]
    if prob > end_prob:
      end_prob = prob
      idx = k

  # backtrack
  seq = []
  for i in range(len_pred-1, -1, -1):
    seq = [idx] + seq
    idx = prev[i][idx]
    
  return seq

def argmax(arr):
  n = len(arr)
  ret = 0
  for i in range(n):
    if arr[i] > arr[ret]:
      ret = i
  return ret

def partition(filename):
  lst = []
  now_lst = []
  f = open(filename, "r")
  for line in f:
    tokens = line.strip().split(' ')
    inst = tokens[0]
    inst_token = inst.split('_')
    if inst_token[2] == "1":
      if len(now_lst) > 0:
        lst.append(now_lst)
      now_lst = [inst_token[0] + "_" + inst_token[1]]
    prob = map(lambda x: math.exp(float(x)), tokens[1:])
    now_lst.append(argmax(prob))
  if len(now_lst) > 0:
    lst.append(now_lst)
  return lst

def window(s):
  ret = ""
  for i in range(4, len(s)):
    tmp = {}
    for j in range(i-4, i+1):
      if s[j] not in tmp:
        tmp[s[j]] = 1
      else:
        tmp[s[j]] += 1
    for j in tmp:
      if tmp[j] >= 3:
        ret += j
  return ret

def trim(s):
  ret = ""
  for i in range(1, len(s)):
    if s[i] != s[i-1]:
      ret += s[i]
  if len(ret) > 1 and ret[0] == 'L':
    ret = ret[1:]
  if len(ret) > 1 and ret[-1] == 'L':
    ret = ret[:-1]
  return ret

def lcs(s1, s2):
  n1 = len(s1)
  n2 = len(s2)
  dp = [[0 for i in range(n2+1)] for i in range(n1+1)]
  prev = [[(0, 0) for i in range(n2+1)] for i in range(n1+1)]
  for i in range(n1):
    for j in range(n2):
      if s1[i] == s2[j]:
        dp[i+1][j+1] = dp[i][j] + 1
        prev[i+1][j+1] = (1, 1)
      elif dp[i+1][j] > dp[i][j+1]:
        dp[i+1][j+1] = dp[i+1][j]
        prev[i+1][j+1] = (0, 1)
      else:
        dp[i+1][j+1] = dp[i][j+1]
        prev[i+1][j+1] = (1, 0)
  x = n1
  y = n2
  ret = ""
  while x > 0 and y > 0:
    tmp = prev[x][y]
    if tmp == (1, 1):
      ret = s1[x-1] + ret
    x -= tmp[0]
    y -= tmp[1]

  return ret

def run():
  lsts = partition("my_test.post")
  f = open("out.csv", "w+")
  f.write("id,phone_sequence\n")
  for lst in lsts:
    print "processing", lst[0], "..."
    seq = viterbi(lst[1:], p_tran, p_emit_2)
    s = []
    for p in seq:
      s += map_phone_to_char[map_48_to_39[map_idx_to_phone[p]]]
    f.write("%s,%s\n" % (lst[0], trim(s)))
  f.close()

build()
run()
