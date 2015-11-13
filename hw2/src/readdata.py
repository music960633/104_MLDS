import re
import numpy
import random

small_train_data_dict = {}

def get_small_train_data(idx):
  global small_train_data_dict
  filename = "../../data/posteriorgram/small_data/train_" \
             + str(idx) \
             + ".post"
  if filename in small_train_data_dict:
    return small_train_data_dict[filename]
  f = open(filename)
  train_inst = []
  train_data = []
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split(' ')
    train_inst += [s[0]]
    train_data += [[float(s[i+1]) for i in range(48)]]
  f.close()
  small_train_data_dict[filename] = (train_inst, train_data)
  return train_inst, train_data

def get_test_post():
  filename = "../../data/posteriorgram/test.post"
  f = open(filename)
  test_inst = []
  test_data = []
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split(' ')
    test_inst += [s[0]]
    test_data += [[float(s[i+1]) for i in range(48)]]
  f.close()
  return test_inst, test_data

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
  filename = "../../data/MLDS_HW1_RELEASE_v1/label/train.lab"
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
