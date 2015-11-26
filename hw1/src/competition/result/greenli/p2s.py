map_48_39 = {}
map_48_char = {}
test_inst = {}
test_data = {}

def get_map_48_39():
  filename = "../../../../../data/MLDS_HW1_RELEASE_v1/phones/48_39.map"
  f = open(filename)
  map_48_39 = {}
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split('\t')
    map_48_39[s[0]] = s[1]
  f.close()
  return map_48_39

def get_map_48_char():
  filename = "../../../../../data/48_idx_chr.map_b"
  f = open(filename)
  map_48_char = {}
  while True:
    s = f.readline()
    if s == "": break
    s = s.strip().split('\t')
    map_48_char[s[0]] = s[1][-1]
  f.close()
  return map_48_char

def match(arr):
  global map_idx_48, map_48_39
  idx = 0
  mx = arr[0]
  for i in range(len(arr)):
    if arr[i] > mx:
      idx = i
      mx = arr[i]
  return map_48_39[map_idx_48[idx]]

def init():
  global map_48_39, map_48_char
  map_48_39   = get_map_48_39()
  map_48_char = get_map_48_char()

def get_test_result():
  global map_48_char, map_48_39
  global test_inst, test_data
  init()
  filename = "./momentum_modified.csv"
  f = open(filename)
  test_inst = []
  test_data = []
  idx = -1
  while True:
    s = f.readline()
    if s == "": break
    temp = s.strip().split(',')
    s = temp[0].strip().split('_')
    if s[-1] == str(1):
      test_inst += [s[0] + "_" + s[1]]
      test_data += [[]]
      idx += 1
    test_data[idx] += [map_48_char[map_48_39[temp[-1]]]]
  f.close()
  return test_inst, test_data

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
  if len(ret) > 1 and ret[0] == 'L':
    ret = ret[1:]
  if len(ret) > 1 and ret[-1] == 'L':
    ret = ret[:-1]
  return ret

def transform():
  global test_inst, test_data
  get_test_result()
  f = open('momentum_rnn_modified.csv', 'w+')
  f.write("id,phone_sequence")
  #for i in range(10):
  for i in range(len(test_inst)):
    f.write("\n%s,%s" %(test_inst[i], trim(test_data[i])))
  f.close()

if __name__ == "__main__":
  transform()
