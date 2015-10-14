import re

f = open("miku.txt", "r")

while True:
   s = f.readline()
   if s == "": break
   tokens = re.findall(r'[.0-9]+', s)
   nums = map(lambda x: float(x), tokens)
   print nums
