#!/usr/bin/env python3

import sys
import numpy as np

import inf_common as IC

if __name__ == "__main__":
  # - open the file which multi_inf_parallel.py outputs
  # (see also the redirect in start.sh), this is argv[1],
  # and plot the development of loss into argv[2]

  times = []
  losses = []
  losses_devs = []
  posrates = []
  posrates_devs = []
  negrates = []
  negrates_devs = []
  
  cnt = 0

  reading = False
  for logname in sys.argv[1:]:
    with open(logname,"r") as f:
      for line in f:
        if line.startswith("Epoch") and "finished at" in line:
          time = int(line.split()[1])
          times.append(time)
        
        # Loss: 1.1441415896882556 +/- 0.1589903560247135
        if line.startswith("Loss:"):
          spl = line.split()
          
          loss = float(spl[1])
          losses.append(loss)
          
          loss_dev = float(spl[-1])
          losses_devs.append(loss_dev)
          
        # Posrate: 0.6208051768416467 +/- 0.28254407289309535
        if line.startswith("Posrate:"):
          spl = line.split()
          rate = float(spl[1])
          posrates.append(rate)
          rate_dev = float(spl[-1])
          posrates_devs.append(rate_dev)
  
        # Negrate: 0.7642507481437519 +/- 0.11107772115788889
        if line.startswith("Negrate:"):
          spl = line.split()
          rate = float(spl[1])
          negrates.append(rate)
          rate_dev = float(spl[-1])
          negrates_devs.append(rate_dev)

    IC.plot_with_devs("{}_deleteme.png".format(sys.argv[1].split("/")[0]),times,losses,losses_devs,posrates,posrates_devs,negrates,negrates_devs)

  idx = np.argmin(losses)
  print("Best loss model",times[idx])
  print("Loss:",losses[idx],"posrate",posrates[idx],"negrate",negrates[idx])

  '''
  print()
  for idx in times:
    if idx % 10 == 0:
      print(idx,losses[idx],posrates[idx],negrates[idx])
  '''

