#!/usr/bin/env python3

import sys
import numpy as np

import inf_common as IC

import matplotlib.pyplot as plt

if __name__ == "__main__":
  # models stat files are simple black spepareted line by line files
  # currently reading:
  # arg1 = "ml_stats.txt": model_idx loss posrate negrate
  # arg2 = "atp_stats.txt": model_idx absolute_num_solved gain_over_the_training_set (aka ATP validation)
  
  ml_stats = {} # model_idx -> (loss, posrate, negrate)
  
  with open(sys.argv[1],"r") as f:
    for line in f:
      idx, loss, posrate, negrate = line.split()
      ml_stats[int(idx)] = (float(loss), float(posrate), float(negrate))

  atp_stats = {} # model_idx -> (solved, gain)

  with open(sys.argv[2],"r") as f:
    for line in f:
      idx, solved, gain = line.split()
      atp_stats[int(idx)] = (float(solved), float(gain))

  indexes_ml = []
  losses = []

  indexes_atp = []
  solveds = []
  gains = []
  losses_atp = []

  for model_idx, (loss, posrate, negrate) in sorted(ml_stats.items()):
    indexes_ml.append(model_idx)
    losses.append(loss)
    if model_idx in atp_stats:
      solved, gain = atp_stats[model_idx]
      
      indexes_atp.append(model_idx)
      solveds.append(solved)
      gains.append(gain)
      losses_atp.append(loss)

  min_solved = 0.0
  # min_solved = np.min(solveds) # so that solveds would not be so high compared to gains

  fig, ax1 = plt.subplots(figsize=(15,10))
  color = 'tab:red'
  ax1.set_xlabel('time (epochs)')
  ax1.set_ylabel('loss', color=color)
  vl, = ax1.plot(indexes_ml, losses, "-", linewidth = 1,label = "loss", color=color)
  # lr, = ax1.plot(times, rates, ":", linewidth = 1,label = "learning_rate", color=color)
  
  ax1.scatter(indexes_atp,losses_atp,marker="x",color = "red")

  ax1.tick_params(axis='y', labelcolor=color)

  # ax1.set_ylim([0.35,0.6])

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('atp-stuff', color=color)  # we already handled the x-label with ax1
  vpr, = ax2.plot(indexes_atp, np.array(solveds)-min_solved, "-", label = "total", color = "blue")
  vnr, = ax2.plot(indexes_atp, gains, "-", label = "gain", color = "cyan")

  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped

  plt.legend(handles = [vl,vpr,vnr], loc='lower left') # loc = 'best' is rumored to be unpredictable

  plt.savefig("stats.png",dpi=250)
  plt.close(fig)

