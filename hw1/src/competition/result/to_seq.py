import sys

def to_seq(infilename, outfilename):
  mp = {}
  fmap = open("../../../../data/48_idx_chr.map_b")
  for line in fmap:
    line = line.strip().replace('\t', ' ').split(' ')
    tokens = [x for x in line if len(x) > 0]
    assert len(tokens) == 3, "parse mapping error"
    mp[tokens[0]] = tokens[2]

  data = []
  fin = open(infilename, "r")
  fout = open(outfilename, "w+")
  fout.write("id,phone_sequence")
  prev = "sil"
  for line in fin:
    line = line.strip()
    if line == "Id,Prediction":
      continue
    tokens = line.split(',')
    assert len(tokens) == 2, "parse error"
    inst_tokens = tokens[0].split('_')
    assert len(inst_tokens) == 3, "instance error"
    if inst_tokens[2] == "1":
      fout.write("\n")
      fout.write(inst_tokens[0] + "_" + inst_tokens[1] + ",")
      prev = "sil"
    if tokens[1] != prev:
      prev = tokens[1]
      fout.write(mp[tokens[1]])
  fout.write("\n")

  fin.close()
  fout.close()

def main():
  if len(sys.argv) != 3:
    print "Usage:", sys.argv[0], "<infile> <outfile>"
    return
  to_seq(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
  main()
