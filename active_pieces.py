#!/usr/bin/env python3

import os,sys
import torch

from collections import defaultdict

import matplotlib.pyplot as plt

if __name__ == "__main__":
  sizes = []
  times = []

  piece_sizes = {} # piece_name -> size
  active = defaultdict(int)

  with open(sys.argv[1],'r') as f:
    for line in f:
      # 0.45389294624328613 starting job on problem piece604.pt of size 22245
      if "starting job on problem" in line:
        spl = line.split()
        piece = spl[-4]
        size = int(spl[-1])

        piece_sizes[piece] = size
        active[piece] += 1

      # 20.660868883132935 job finished at on problem piece4275.pt started 3.1876797676086426 finished 20.65919327735901 took 17.471513509750366
      if "job finished at on problem" in line:
        spl = line.split()

        took = float(spl[-1])
        piece = spl[6]

        size = piece_sizes[piece] # it must be there!

        sizes.append(size)
        times.append(took)

        active[piece] -= 1

  cnt = 0
  total = 0
  for piece,num in sorted(active.items(),key = lambda x : int(x[0][5:-3])):
    for _ in range(num):
      size = piece_sizes[piece]
      print(piece,size)
      cnt += 1
      total += size

  print("Total",cnt,"pieces of accummulated size",total)

  fig, ax1 = plt.subplots()
  ax1.set_xlabel('size')
  ax1.set_ylabel('time')

  ax1.scatter(sizes,times,marker="x")
  plt.savefig("sizes_times.png",dpi=250)
  plt.close(fig)
