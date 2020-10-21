#!/usr/bin/env python3

import inf_common as IC

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools,os

import numpy as np

from multiprocessing import Pool

import matplotlib.pyplot as plt

def eval_one(task):
  (i,epoch_model_file,blob) = task
  
  print(i,epoch_model_file,flush=True)
  
  (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) = blob
  
  parts = torch.load(epoch_model_file)
  
  model = IC.LearningModel(*parts,init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,save_logits=True)
  model.train()
  model.forward()
  
  logits_dict = model.logits
  
  return (i,logits_dict)

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
  # To be called as in: ./blob_trajectory.py mizarEasy/abstractOnly/big_blob.pt mizarEasy/abstractOnly/run_p2.0/
  #
  # firs is a file with the blob, second is the folder which contains the model-epoch*.pt files

  blob = torch.load(sys.argv[1])
  (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) = blob
  
  size_min = sys.float_info.max
  size_max = sys.float_info.min
  
  for id in sorted(set(pos_vals) | set(neg_vals)):
    pos = pos_vals[id]
    neg = neg_vals[id]
  
    size = pos+neg
    if size > size_max:
      size_max = size
    if size < size_min:
      size_min = size

  print(size_min,size_max)

  tasks = []
  for i in itertools.count():
    epoch_model_file = "{}/model-epoch{}.pt".format(sys.argv[2],i)
    if not os.path.exists(epoch_model_file):
      break
    tasks.append((i,epoch_model_file,blob))

  '''
  results = []
  for task in tasks:
    res = eval_one(task)
    results.append(res)
  '''
  pool = Pool(processes=5)
  results = pool.map(eval_one, tasks, chunksize = 1)
  pool.close()
  pool.join()
  del pool

  print("Plotting",flush=True)

  fig, ax = plt.subplots(figsize=(20, len(results)))

  ax.set_xlabel('logit value')
  ax.set_ylabel('epoch')

  for id in sorted(set(pos_vals) | set(neg_vals)):
    pos = pos_vals[id]
    neg = neg_vals[id]
    
    size = (pos+neg)
    color = pos/size
    
    if id > 2048:
      break
    epochs = []
    logits = []
    for (i,logits_dict) in sorted(results):
      epochs.append(i)
      logits.append(logits_dict[id])

    print(id,color,size,(size-size_min)/(size_max-size_min),flush=True)

    something, = ax.plot(logits, epochs, "-", linewidth = 1, label = "cl{}".format(id), color = "gray", zorder=1)
    ax.scatter(logits, epochs, c = [color]*len(epochs), s = 20+200*(size-size_min)/(size_max-size_min), vmin = 0.0, vmax = 1.0, zorder=2)


  # fig.colorbar()

  # plt.legend(handles = [lnv,lps,lpv], loc='upper left') # loc = 'best' is rumored to be unpredictable

  # plt.show()

  filename = "blob_trajectory.png"
  plt.savefig(filename,dpi=250)
  print("Saved final plot to",filename)
  plt.close(fig)


