def smooth(infilename, outfilename):
  fin = open(infilename, "r")
  data = []
  for line in fin:
    line = line.strip()
    if line == "Id,Prediction":
      continue
    tokens = line.split(',')
    assert len(tokens) == 2, "parse error"
    data.append((tokens[0], tokens[1]))
  fin.close()

  fout = open(outfilename, "w+")
  fout.write("Id,Prediction\n")
  for i in range(len(data)):
    fout.write(data[i][0] + ",")
    if i > 0 and data[i][1] == data[i-1][1]:
      phone = data[i][1]
    elif i < len(data)-1 and data[i][1] == data[i+1][1]:
      phone = data[i][1]
    elif i > 0:
      phone = data[i-1][1]
    else:
      phone = data[i+1][1]
    fout.write(phone + "\n")
  fout.close()

smooth("result/new.csv", "result/smooth.csv")
