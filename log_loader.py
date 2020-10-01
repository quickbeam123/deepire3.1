#!/usr/bin/env python3

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

import inf_common as IC

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Load a set of logs, passed as arguments, each will become a "training example tree"
  # Scan the set and create a histogram (to learn which initial and derivational networks will be needed; including the "dropout" defaults)
  # - save it to "data_hist.pt"
  # normalize the training data and save that to "training_data.pt"
  #
  # actually, all the logs to read need to be saved in a file an passed as sys.argv[3]
  #
  # To be called as in: ./log_loader.py data_hist.pt training_data.pt *.log-files-listed-line-by-line-in-a-file (an "-s4k on" run of vampire)

  prob_data = {} # probname -> (init,deriv,pars,selec,good)
  
  with open(sys.argv[3],"r") as f:
    for line in f:
      probname = line[:-1]
      probdata = IC.load_one(probname)
      if probdata is not None: # None when the problem was Saturated / Satisfiable
        prob_data[probname] = probdata

  print(len(prob_data),"problems loaded!")
  print()

  init_hist,deriv_hist = IC.prepare_hists(prob_data)

  print("init_hist",init_hist)
  print("deriv_hist",deriv_hist)
  print("Saving hist to",sys.argv[1])
  torch.save((init_hist,deriv_hist), sys.argv[1])
  print()

  # NORMALIZE PROBLEM DATA:
  # 1) it's better to have them in a list (for random.choice)
  # 2) it's better to disambiguate clause indices, so that any union of problems will make sense as on big graph
  prob_data_list = IC.normalize_prob_data(prob_data)

  print("prob_data_list normalized")
  print("Saving prob_data_list to",sys.argv[2])
  torch.save(prob_data_list, sys.argv[2])
  print("Done")
