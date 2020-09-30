#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  # - open the file which multi_inf_parallel.py outputs
  # (see also the redirect in start.sh), this is argv[1],
  # and plot the development of loss into argv[3],
  # only start "recording" when the loss drops below argv[2]

  points = []
  reading = False
  with open(sys.argv[1],"r") as f:
    for line in f:
      if line.startswith("Global: "):
        spl = line.split()
        pnt = float(spl[1])
        if reading or pnt < float(sys.argv[2]):
          reading = True
          points.append(pnt)

  plt.plot(points)
  plt.savefig(sys.argv[3])
