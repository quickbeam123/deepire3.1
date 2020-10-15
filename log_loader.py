#!/usr/bin/env python3

import inf_common as IC

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
  probdata = IC.load_one(probname,max_size=15000)
  print("Took",time.time()-start_time)
  return probdata

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
  # data_hist.pt training_data.pt validation_data.pt are created in <folder>

  prob_data_list = [] # [(probname,(init,deriv,pars,selec,good)]

  '''
  tasks = []
  with open(sys.argv[2],"r") as f:
    for i,line in enumerate(f):
      if i >= 1000:
        break
      probname = line[:-1]
      tasks.append((i,probname))
  pool = Pool(processes=50)
  results = pool.map(load_one, tasks, chunksize = 100)
  prob_data_list = list(filter(None, results))
  '''
  
  prob_data_list = []
  with open(sys.argv[2],"r") as f:
    for i,line in enumerate(f):
      probname = line[:-1]
      probdata = load_one((i,probname))
      if probdata is not None:
        prob_data_list.append((probname,probdata))
      if len(prob_data_list) >= 1000:
        break

  print(len(prob_data_list),"problems loaded!")
  print()

  '''
  filename = "{}/raw_prob_data_list.pt".format(sys.argv[1])
  torch.save(prob_data_list, filename)
  '''
  
  '''
  filename = "{}/raw_prob_data_list.pt".format(sys.argv[1])
  prob_data_list = torch.load(filename)
  '''

  init_sign,deriv_arits,axiom_hist = IC.prepare_signature(prob_data_list)

  if True: # We want to use axiom names rather than theory_axiom ids:
    init_sign,prob_data_list,thax_to_str = IC.axiom_names_instead_of_thax(init_sign,axiom_hist,prob_data_list)
  else:
    thax_to_str = {}

  print("init_sign",init_sign)
  print("deriv_arits",deriv_arits)
  #print("axiom_hist",axiom_hist)
  print("thax_to_str",thax_to_str)

  filename = "{}/data_sign.pt".format(sys.argv[1])
  print("Saving singature to",filename)
  torch.save((init_sign,deriv_arits,thax_to_str), filename)
  print()

  print("Compressing (and dropping info about axioms)")
  for i, (probname,(init,deriv,pars,selec,good,axioms)) in enumerate(prob_data_list):
    print(probname,"init: {}, deriv: {}, select: {}, good: {}".format(len(init),len(deriv),len(selec),len(good)))
    prob_data_list[i] = IC.compress_prob_data([(probname,(init,deriv,pars,selec,good))])
  print()

  random.shuffle(prob_data_list)
  spl = int(len(prob_data_list) * 0.8)
  print("shuffled and split at idx",spl,"out of",len(prob_data_list))

  filename = "{}/training_data.pt".format(sys.argv[1])
  print("Saving training part to",filename)
  torch.save(prob_data_list[:spl], filename)
  filename = "{}/validation_data.pt".format(sys.argv[1])
  print("Saving testing part to",filename)
  torch.save(prob_data_list[spl:], filename)

  print("Done")
