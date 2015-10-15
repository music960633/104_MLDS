#! /usr/bin/python

import Image
import re
import sys

def main():
  if len(sys.argv) != 3:
    print "usage:", sys.argv[0], "<input> <output>"
    return
  img = Image.new("RGB", (500, 500))
  pixels = img.load()


  f = open(sys.argv[1])
  while True:
    s = f.readline()
    if s == "": break
    nums = map(lambda x: float(x), re.findall(r'[-.0-9]+', s))
    x = int(nums[0] * 100 + 250)
    y = int(nums[1] * 100 + 250)
    pixels[x, y] = (100, 100, 100) if nums[2] == 1 else (0, 0, 0)

  img.save(sys.argv[2], "PNG")

if __name__ == "__main__":
  main()
