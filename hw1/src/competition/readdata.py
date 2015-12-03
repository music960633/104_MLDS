import re
import numpy
import random

N_INPUT = 69

def get_fbank(filename):
  f = open(filename)
  inst = []
  fbank = [[] for i in range(N_INPUT)]
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split(' ')
    inst += [s[0]]
    for i in range(N_INPUT):
      fbank[i] += [float(s[i+1])]
  f.close()
  return inst, fbank

def get_train_fbank():
  filename = "../../../data/MLDS_HW1_RELEASE_v1/fbank/small_data/train_all.ark"
  return get_fbank(filename)

def get_seg_train_fbank(idx):
  filename = "../../../data/MLDS_HW1_RELEASE_v1/fbank/train2_" \
             + str(idx) + ".ark"
  return get_fbank(filename)

def get_validate_fbank(idx):
  filename = "../../../data/MLDS_HW1_RELEASE_v1/fbank/small_data/validate_" \
             + str(idx) + ".ark"
  return get_fbank(filename)

def get_test_fbank():
  filename = "../../../data/MLDS_HW1_RELEASE_v1/fbank/test2.ark"
  return get_fbank(filename)

def get_map_48_39():
  filename = "../../../data/MLDS_HW1_RELEASE_v1/phones/48_39.map"
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
  filename = "../../../data/MLDS_HW1_RELEASE_v1/label/train.lab"
  f = open(filename)
  map_inst_48 = {}
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split(',')
    map_inst_48[s[0]] = s[1]
  f.close()
  return map_inst_48

def get_map_48_idx():
  filename = "../../../data/MLDS_HW1_RELEASE_v1/phones/48_39.map"
  f = open(filename)
  map_48_idx = {}
  idx = 0
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split('\t')
    map_48_idx[s[0]] = idx
    idx += 1
  f.close()
  return map_48_idx
