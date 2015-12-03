#f1 = open("./rnn_input", "r")
#f2 = open("../MLDS_HW1_RELEASE_v1/label/train.lab", "r")
f1 = open("./rnn_test", "r")
f2 = open("./my_test.post", "r")
myfile = open("./my_test_input", "w")
while True:
  #for i in range(3):
  s1 = f1.readline()
  s2 = f2.readline()
  if s1 == "" or s2 == "": break
  s1 = s1.strip().split(' ')
  #s2 = s1.strip().split(',')
  s2 = s2.strip().split(' ')  
  myfile.write("%s" %(s2[0]))
  for i in range(48):
    myfile.write(" %f" %float(s1[i]))
    if i == 47: myfile.write("\n")
