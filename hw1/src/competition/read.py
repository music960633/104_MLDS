import re
import numpy

def get_fbank():
  filename = "../../data/MLDS_HW1_RELEASE_v1/fbank/train.ark"
  f = open(filename)
  map_fbank = {}
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split(' ')
    map_fbank[s[0]] = [numpy.float32(x) for x in s[1:]]
  f.close()
  return map_fbank

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


mapping = get_fbank()
# print mapping.items()
