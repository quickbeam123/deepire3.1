#!/usr/bin/env python3

import sys
import numpy as np

import inf_common as IC

if __name__ == "__main__":
  # - open the file which multi_inf_parallel.py outputs
  # (see also the redirect in start.sh), this is argv[1],
  # and plot the development of loss into argv[2]

  times = []
  train_losses = []
  train_posrates = []
  train_negrates = []
  valid_losses = []
  valid_posrates = []
  valid_negrates = []
  
  cnt = 0

  reading = False
  for logname in sys.argv[1:]:
    with open(logname,"r") as f:
      for line in f:
        if line.startswith("Epoch") and "training finished at" in line:
          time = int(line.split()[1])
          times.append(time)
        
        if line.startswith("Training stats:"):
          spl = line.split()
          loss = float(spl[2])
          posrate = float(spl[3])
          negrate = float(spl[4])
          
          train_losses.append(loss)
          train_posrates.append(posrate)
          train_negrates.append(negrate)
        
        if line.startswith("Validation stats:"):
          spl = line.split()
          loss = float(spl[2])
          posrate = float(spl[3])
          negrate = float(spl[4])
          
          valid_losses.append(loss)
          valid_posrates.append(posrate)
          valid_negrates.append(negrate)

    IC.plot_one("deleteme.png",times,train_losses,train_posrates,train_negrates,valid_losses,valid_posrates,valid_negrates)

  idx = np.argmin(train_losses)
  print("Best train loss model",times[idx])
  print("Loss:",train_losses[idx],"posrate",train_posrates[idx],"negrate",train_negrates[idx])
  idx = np.nanargmin(valid_losses)
  print("Best valid loss model",times[idx])
  print("Loss:",valid_losses[idx],"posrate",valid_posrates[idx],"negrate",valid_negrates[idx])


