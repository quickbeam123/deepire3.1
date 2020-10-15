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
  # Load data_hist.pt and trained model; export a torchscript model version of it to argv[3]
  #
  # To be called as in: ./exporter.py enigma_smt_447/data_sign.pt enigma_smt_447/models14/inf_14_Tanh_p0.9791753101758176_n0.5020857886700685.pt enigma_smt_447/model_14Tanh_best.pt

  # inf_41_Tanh_p0.9905907013270361_n0.6047052650764457.pt

  init_sign,deriv_arits,thax_to_str = torch.load(sys.argv[1])
  print("Loaded signature from",sys.argv[1])

  IC.create_saver(init_sign,deriv_arits,thax_to_str)
  import inf_saver as IS

  parts = torch.load(sys.argv[2])
  parts_copies = torch.load(sys.argv[2])

  print("Loaded model from",sys.argv[2])

  IS.save_net(sys.argv[3],parts,parts_copies)

  print("Exported to",sys.argv[3])

