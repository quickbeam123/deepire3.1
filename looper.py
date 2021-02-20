#!/usr/bin/env python3

# load inf_common before torch, so that torch is single threaded
import inf_common as IC

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # To be called as in: ./looper.py TODO

  train_data_idx = torch.load("{}/training_index.pt".format(sys.argv[1]))
  print("Loaded train data:",len(train_data_idx))
  valid_data_idx = torch.load("{}/validation_index.pt".format(sys.argv[1]))
  print("Loaded valid data:",len(valid_data_idx))
  
  size_total = 0
  for (size,piece_name) in train_data_idx:
    #print(piece_name,size)
    size_total += size
  for (size,piece_name) in valid_data_idx:
    # print(piece_name,size)
    size_total += size

  print("size_total",size_total)

  exit(0)


