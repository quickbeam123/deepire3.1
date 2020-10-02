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
  # Load data_hist.pt and create a fresh model save it to "model_name_prefix"+characterization
  #
  # To be called as in: ./initializer.py enigma_smt_447/data_hist.pt enigma_smt_447/model0

  init_hist,deriv_hist = torch.load(sys.argv[1])
  print("Loaded hist from",sys.argv[1])

  model0 = IC.get_initial_model(init_hist,deriv_hist)

  model_name = sys.argv[2]+IC.name_initial_model_suffix()
  torch.save(model0,model_name)
  print("New model saved to",model_name)
