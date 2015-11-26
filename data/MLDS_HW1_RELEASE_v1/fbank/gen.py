#! /usr/bin/python

import random

def gen_train_fbank():
  filename = "train.ark"
  # filename = "train.ark"
  f = open(filename, "r")
  data = []
  while True:
    s = f.readline()
    if s == "": break
    data += [s]
  f.close()
  data_size = len(data)

  for i in range(100):
    filename = "small_data/train_" + str(i) + ".ark"
    print "generating", filename, "..."
    f = open(filename, "w+")
    for j in range(100000):
      idx = int(random.random() * data_size)
      f.write(data[idx])
    f.close()

if __name__ == "__main__":
  gen_train_fbank()
