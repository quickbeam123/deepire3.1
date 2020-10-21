#!/usr/bin/env python3

import inf_common as IC

import hyperparams as HP

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

from multiprocessing import Pool

def load_one(task):
  i,probname = task
  
  print(i)
  start_time = time.time()
  probdata = IC.load_one(probname) # ,max_size=15000)
  print("Took",time.time()-start_time)
  if probdata:
    return probname,probdata
  else:
    None

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Load a set of logs, passed in a file as the final argument, each will become a "training example tree"
  # Scan the set and create a histogram (to learn which initial and derivational networks will be needed; including the "dropout" defaults)
  # - save it to "data_hist.pt"
  # normalize the training data and save that to "training_data.pt" / "validation_data.pt" using a 80:20 split
  #
  # To be called as in: ./log_loader.py <folder> *.log-files-listed-line-by-line-in-a-file (an "-s4k on" run of vampire)
  #
  # data_sign.pt and raw_log_data_*.pt are created in <folder>

  prob_data_list = [] # [(probname,(init,deriv,pars,selec,good)]

  tasks = []
  with open(sys.argv[2],"r") as f:
    for i,line in enumerate(f):
      """
      if i >= 1000:
        break
      """
      probname = line[:-1]
      tasks.append((i,probname))
  pool = Pool(processes=50)
  results = pool.map(load_one, tasks, chunksize = 100)
  pool.close()
  pool.join()
  del pool
  prob_data_list = list(filter(None, results))

  '''
  prob_data_list = []
  with open(sys.argv[2],"r") as f:
    for i,line in enumerate(f):
      probname = line[:-1]
      result = load_one((i,probname))
      if result is not None:
        prob_data_list.append(result)
      
      """
      if len(prob_data_list) >= 1000:
        break
      """
  '''

  print(len(prob_data_list),"problems loaded!")

  thax_sign,sine_sign,deriv_arits,axiom_hist = IC.prepare_signature(prob_data_list)

  if HP.THAX_SOURCE == HP.ThaxSource_AXIOM_NAMES: # We want to use axiom names rather than theory_axiom ids:
    thax_sign,prob_data_list,thax_to_str = IC.axiom_names_instead_of_thax(thax_sign,axiom_hist,prob_data_list,axcnt_cutoff=HP.AXCNT_CUTOFF)
  else:
    thax_to_str = {}

  print("thax_sign",thax_sign)
  print("sine_sign",sine_sign)
  print("deriv_arits",deriv_arits)
  #print("axiom_hist",axiom_hist)
  print("thax_to_str",thax_to_str)

  filename = "{}/data_sign.pt".format(sys.argv[1])
  print("Saving singature to",filename)
  torch.save((thax_sign,sine_sign,deriv_arits,thax_to_str), filename)
  print()

  filename = "{}/raw_log_data{}".format(sys.argv[1],IC.name_raw_data_suffix())
  print("Saving raw data to",filename)
  torch.save(prob_data_list, filename)
  print()

  print("Done")

