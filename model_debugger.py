#!/usr/bin/env python3

import inf_common as IC

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

import numpy as np

from multiprocessing import Pool

import matplotlib.pyplot as plt

# release the memory just claimed - an experiment
import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

def contribute(id,model,selec,good,posOK,posTot,negOK,negTot,pos_cuts,neg_cuts,min_pos_logit):
  howmuch = 1.0/len(selec)
  
  # print(id,thax,val,id in selec,id in good)
  logit = model(id) # calling forward
  # print(id,thax,logit,id in selec,id in good)
  val = (logit >= 0.0) # interpreting the logit

  if id in selec:
    if id in good:
      pos_cuts[logit] += howmuch
      posTot += howmuch
      if val:
        posOK += howmuch
      
      if logit < min_pos_logit:
        min_pos_logit = logit
  
    else:
      neg_cuts[logit] += howmuch
      negTot += howmuch
      if not val:
        negOK += howmuch

  return posOK,posTot,negOK,negTot,min_pos_logit

def eval_one(task):
  (metainfo,(init,deriv,pars,selec,good,axioms)) = task
  print(metainfo)
  model = torch.jit.load(sys.argv[2]) # always load a new model -- it contains the lookup tables for the particular model

  posOK = 0.0
  posTot = 0.0
  negOK = 0.0
  negTot = 0.0
  
  # keep collecting the logis values at which we would need to cut to get this clause classified as positive
  pos_cuts = defaultdict(int)
  neg_cuts = defaultdict(int)
  
  min_pos_logit = sys.float_info.max

  for id, (thax,sine) in init:
    if thax == -1:
      st = "-1"
    elif id in axioms:
      st = axioms[id]
    else:
      st = str(thax)
    
    # communication via st and sine
    getattr(model,"new_init")(id,[-1,-1,-1,-1,-1,sine],st)

    posOK,posTot,negOK,negTot,min_pos_logit = contribute(id,model,selec,good,posOK,posTot,negOK,negTot,pos_cuts,neg_cuts,min_pos_logit)

  for id, (rule) in deriv:
    if rule == 666:
      my_pars = pars[id]
      assert(len(my_pars) == 1)
      getattr(model,"new_avat")(id,[-1,-1,-1,my_pars[0]])
    else:
      getattr(model,"new_deriv{}".format(rule))(id,[-1,-1,-1,-1,rule],pars[id])

    posOK,posTot,negOK,negTot,min_pos_logit = contribute(id,model,selec,good,posOK,posTot,negOK,negTot,pos_cuts,neg_cuts,min_pos_logit)

  del model
  libc.malloc_trim(ctypes.c_int(0))

  return (metainfo,posOK,posTot,negOK,negTot,pos_cuts,neg_cuts,min_pos_logit)

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Load a torchscript model and a set of logs, passed in a file as the final argument,
  # test the model on the logs (as if vampire was running) and report individual and average pos/neg rates
  #
  # To be called as in: ./model_debugger.py raw_log_data_*.pt torch_script_model.pt

  prob_data_list = torch.load(sys.argv[1])
  
  '''
  results = []
  for task in prob_data_list:
    res = eval_one(task)
    print("Done",task[0])
    results.append(res)
  '''
  pool = Pool(processes=25)
  results = pool.map(eval_one, prob_data_list, chunksize = 5)
  pool.close()
  pool.join()
  del pool

  cnt = 0
  posrate_sum = 0.0
  negrate_sum = 0.0
  
  per_prob = defaultdict(list)
  pos_cuts_fin = defaultdict(int)
  neg_cuts_fin = defaultdict(int)

  for (probname,posOK,posTot,negOK,negTot,pos_cuts,neg_cuts,min_pos_logit) in results:
    
    posrate = posOK / posTot if posTot > 0 else 1.0
    negrate = negOK / negTot if negTot > 0 else 1.0
    
    cnt += 1
    print(probname)
    print(cnt,"Posrate",posrate,"negrate",negrate)
  
    posrate_sum += posrate
    negrate_sum += negrate

    for logit, val in pos_cuts.items():
      pos_cuts_fin[logit] += val
    
    for logit, val in neg_cuts.items():
      neg_cuts_fin[logit] += val

    per_prob[min_pos_logit].append(probname)

  print()
  print("Total probs:",cnt)
  print("Final posrate:",posrate_sum/cnt)
  print("Final negrate:",negrate_sum/cnt)

  print()
  print("per_prob pos example logit minima")
  for logit, problems in sorted(per_prob.items()):
    print(logit,len(problems))

  print()
  print("Plotting")

  logits = []
  cur_pos = 0
  pos_vals = []
  cur_neg = 0
  neg_vals = []
  cur_prob = 0
  
  prob_logits = []
  prob_vals = []
  for logit in sorted(set(pos_cuts_fin) | set(neg_cuts_fin)):
    logits.append(logit)
    cur_pos -= pos_cuts_fin[logit]
    pos_vals.append(cur_pos)
    cur_neg += neg_cuts_fin[logit]
    neg_vals.append(cur_neg)
    
    if per_prob[logit]:
      prob_logits.append(logit)
      cur_prob += len(per_prob[logit])
      prob_vals.append(cur_prob)

  fig, ax1 = plt.subplots(figsize=(8, 6))

  color = 'tab:blue'
  ax1.set_xlabel('logit value')
  ax1.set_ylabel('clause count (%)', color=color)
  lnv, = ax1.plot(logits, np.array(neg_vals)/neg_vals[-1], ".", linewidth = 1, label = "neg_vals", color="green")
  lps, = ax1.plot(logits, (np.array(pos_vals)-pos_vals[-1])/(-pos_vals[-1]), ".", linewidth = 1,label = "pos_vals", color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax1.axhline(0.0)
  ax1.axhline(1.0)

  ax1.axvline(0.0)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:red'
  ax2.set_ylabel('problem %', color=color)  # we already handled the x-label with ax1

  lpv, = ax2.plot(prob_logits, np.array(prob_vals)/prob_vals[-1], "+", label = "per_prob_pos_min", color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  low1,high1 = ax1.get_ylim()
  low2,high2 = ax2.get_ylim()

  # print(low1,high1)
  # print(low2,high2)
  # ax2.set_ylim((-0.5,high2))

  fig.tight_layout()  # otherwise the right y-label is slightly clipped

  plt.legend(handles = [lnv,lps,lpv], loc='upper left') # loc = 'best' is rumored to be unpredictable

  # plt.show()

  filename = "debugger_plot_{}.png".format(sys.argv[2].split("/")[-1])
  plt.savefig(filename,dpi=250)
  print("Saved final plot to",filename)
  plt.close(fig)


