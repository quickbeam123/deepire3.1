#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  # - open the file which multi_inf_parallel.py outputs
  # (see also the redirect in start.sh), this is argv[1],
  # and plot the development of loss into argv[3],
  # only start "recording" when the loss drops below argv[2]

  losses = []
  posrates = []
  negrates = []
  
  cnt = 0
  
  reading = False
  with open(sys.argv[1],"r") as f:
    for line in f:
      if line.startswith("Global: "):
        spl = line.split()
        loss = float(spl[1])
        posrate = float(spl[2])
        negrate = float(spl[3])
        
        cnt += 1
        if cnt % 100 != 0:
          continue
        
        if reading or loss < float(sys.argv[2]):
          reading = True
          losses.append(loss)
          posrates.append(posrate)
          negrates.append(negrate)

    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('time (epochs)')
    ax1.set_ylabel('loss', color=color)
    tl, = ax1.plot(losses, "-", linewidth = 1,label = "train_loss", color=color)
    vl, = ax1.plot(np.array(losses)-0.3, "--", linewidth = 1, label = "valid-loss", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('pos/neg-rate', color=color)  # we already handled the x-label with ax1
    tpr, = ax2.plot(posrates, "-", label = "posrate", color = "blue")
    tnr, = ax2.plot(negrates, "--", label = "negrate", color = "cyan")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.legend(handles = [tl,vl,tpr,tnr], loc='lower left')
    plt.savefig(sys.argv[3],dpi=250)
