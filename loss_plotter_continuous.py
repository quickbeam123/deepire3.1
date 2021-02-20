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
  rates = []
  last_rate = None
  
  cnt = 0

  reading = False
  for logname in sys.argv[1:]:
    with open(logname,"r") as f:
      for line in f:
        if line.startswith("Epoch") and "finished at" in line:
          time = int(line.split()[1])
          # print("Have epoch",time)
          times.append(time)
          rates.append(last_rate)
        
        # Loss: 1.1441415896882556 +/- 0.1589903560247135
        if line.startswith("Loss:"):
          spl = line.split()
          
          loss = float(spl[1])
          losses.append(loss)
          
          # print("Have loss",loss)
          
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
  
        if "effective" in line:
          last_rate = float(line.split()[-1])

  # the normal mode:
  if False:
    IC.plot_with_devs("{}_deleteme.png".format(sys.argv[1].split("/")[0]),times,losses,losses_devs,posrates,posrates_devs,negrates,negrates_devs)

  # hack one for the cade paper
  if False:
    IC.plot_with_devs_just_loss_and_LR("run40_training.png",times,losses,losses_devs,rates,clipLoss=[0.37,1.43])

  # hack two for the cade paper
  if True:
    atp_models = []
    atp_gains = []
    
    with open("smtlibbing/atp_stats.txt","r") as f:
      for line in f:
        idx, solved, gain = line.split()
        atp_models.append(int(idx))
        atp_gains.append(int(gain))

    IC.plot_with_devs_just_loss_and_ATPeval("run40_validation.png",times,losses,losses_devs,atp_models,atp_gains,clipLoss=[0.37,1.43])

  idx = np.argmin(losses)
  print("Best loss model",times[idx])
  print("Loss:",losses[idx],"posrate",posrates[idx],"negrate",negrates[idx])

  print()
  for idx,nominal_idx in enumerate(times):
    #if idx % 10 == 0:
    print(nominal_idx,losses[idx],posrates[idx],negrates[idx])

