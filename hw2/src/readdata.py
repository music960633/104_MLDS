import re
import numpy
import random

def get_train_post():
  filename = "../../data/posteriorgram/train1.post"
  # filename = "../../data/MLDS_HW1_RELEASE_v1/fbank/train.ark"
  f = open(filename)
  train_inst = []
  train_fbank = [[] for i in range(48)]
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split(' ')
    train_inst += [s[0]]
    for i in range(48):
      train_fbank[i] += [float(s[i+1])]
  f.close()
  return train_inst, train_fbank

def get_test_post():
  filename = "../../data/posteriorgram/test1.post"
  # filename = "test.ark"
  f = open(filename)
  test_inst = []
  test_fbank = [[] for i in range(48)]
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split(' ')
    test_inst += [s[0]]
    for i in range(48):
      test_fbank[i] += [float(s[i+1])]
  f.close()
  return test_inst, test_fbank

def get_map_48_39():
  filename = "../../data/MLDS_HW1_RELEASE_v1/phones/48_39.map"
  f = open(filename)
  map_48_39 = {}
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split('\t')
    map_48_39[s[0]] = s[1]
  f.close()
  return map_48_39

def get_map_inst_48():
  filename = "../../data/MLDS_HW1_RELEASE_v1/label/train1.lab"
  f = open(filename)
  map_inst_48 = {}
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split(',')
    map_inst_48[s[0]] = s[1]
  f.close()
  return map_inst_48

def get_map_48_char():
  filename = "../../data/48_idx_chr.map_b"
  f = open(filename)
  map_48_char = {}
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split('\t')
    map_48_char[s[0]] = s[1][-1]
  f.close()
  return map_48_char
