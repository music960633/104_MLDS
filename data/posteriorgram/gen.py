#! /usr/bin/python

import random

def gen_train_fbank():
  f = open("my_train.post", "r")
  data = []
  i = 0
  filename = "small_data/train_" + str(i) + ".post"
  small_file = open(filename, "w+")
  while True:
    s = f.readline()
    if s == "": break
    temp = s.strip().split(' ')
    temp = temp[0].strip().split('_')
    if int(temp[-1]) % 100 == 1:
      print "generating train_" + str(i) + ".post"
      if i != 0:
        small_file.close()
        filename = "small_data/train_" + str(i) + ".post"
        small_file = open(filename, "w+")
      i = i + 1
    small_file.write(s)
  f.close()

def gen_test_fbank():
  f = open("my_test.post", "r")
  data = []
  i = 0
  filename = "small_test/test_" + str(i) + ".post"
  small_file = open(filename, "w+")
  while True:
    s = f.readline()
    if s == "": break
    temp = s.strip().split(' ')
    temp = temp[0].strip().split('_')
    if temp[-1] == '1':
      print "generating test_" + str(i) + ".post"
      if i != 0:
        small_file.close()
        filename = "small_test/test_" + str(i) + ".post"
        small_file = open(filename, "w+")
      i = i + 1
    small_file.write(s)
  f.close()

if __name__ == "__main__":
  gen_train_fbank()
  gen_test_fbank()
