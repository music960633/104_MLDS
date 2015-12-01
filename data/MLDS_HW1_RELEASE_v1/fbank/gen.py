#! /usr/bin/python

import random

def gen_small_train_fbank():
  filename = "train.ark"
  # filename = "train.ark"
  f = open(filename, "r")
  data = []
  for s in f:
    tokens = s.strip().split(' ')
    assert len(tokens) == 70
    data.append((tokens[0], tokens[1:]))
  f.close()
  data_size = len(data)

  for i in range(10):
    filename = "small_data/train_" + str(i) + ".ark"
    print "generating", filename, "..."
    f = open(filename, "w+")
    for j in range(30000):
      idx = int(random.random() * data_size)
      f.write(data[idx][0])
      for j in range(idx-1, idx+1+1):
        if j < 0 or j >= data_size:
          for k in range(69):
            f.write(" 0.0")
        else:
          for k in data[j][1]:
            f.write(" " + k)
      f.write('\n')
    f.close()

def gen_test_fbank():
  f = open("test.ark", "r")
  data = []
  for s in f:
    tokens = s.strip().split(' ')
    assert len(tokens) == 70
    data.append((tokens[0], tokens[1:]))
  f.close()
  data_size = len(data)

  f = open("test2.ark", "w+")
  for i in range(data_size):
    f.write(data[i][0])
    for j in range(i-1, i+1+1):
      if j < 0 or j >= data_size:
        for k in range(69):
          f.write(" 0.0")
      else:
        for k in data[j][1]:
          f.write(" " + k)
    f.write('\n')
  f.close()

def gen_train_fbank():
  f = open("train.ark", "r")
  data = []
  for s in f:
    tokens = s.strip().split(' ')
    assert len(tokens) == 70
    data.append((tokens[0], tokens[1:]))
  f.close()
  data_size = len(data)

  idx = 0
  cnt = 0
  f = open("train2_" + str(idx) + ".ark", "w+")
  for i in range(data_size):
    f.write(data[i][0])
    for j in range(i-1, i+1+1):
      if j < 0 or j >= data_size:
        for k in range(69):
          f.write(" 0.0")
      else:
        for k in data[j][1]:
          f.write(" " + k)
    f.write('\n')
    cnt += 1
    if cnt > 100000:
      f.close()
      idx += 1
      cnt = 0
      f = open("train2_" + str(idx) + ".ark", "w+")

  f.close()

if __name__ == "__main__":
  print 1
  gen_small_train_fbank()
  print 2
  gen_test_fbank()
  print 3
  gen_train_fbank()
