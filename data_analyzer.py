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
  # To be called as in: ./data_analyzer.py training_data.pt

  prob_data_list = torch.load(sys.argv[1]) # [ probname, (init,deriv,pars,selec,good)]
  
  '''
  i = 208

  (probname,(init,deriv,pars,selec,good)) = prob_data_list[i]
  print(i,probname,len(init),len(deriv),len(pars),len(selec),len(good))
  (probname,(init,deriv,pars,selec,good)) = IC.compress_prob_data([prob_data_list[i]])
  print(i,probname,len(init),len(deriv),len(pars),len(selec),len(good))
  print()
  
  i = 662
  
  (probname,(init,deriv,pars,selec,good)) = prob_data_list[i]
  print(i,probname,len(init),len(deriv),len(pars),len(selec),len(good))
  (probname,(init,deriv,pars,selec,good)) = IC.compress_prob_data([prob_data_list[i]])
  print(i,probname,len(init),len(deriv),len(pars),len(selec),len(good))
  print()
  
  (probname,(init,deriv,pars,selec,good)) = IC.compress_prob_data([prob_data_list[208],prob_data_list[662]])
  print(i,probname,len(init),len(deriv),len(pars),len(selec),len(good))
  print()
  
  exit(0)
  '''
  
  init_len_hist = defaultdict(int)
  sel_len_hist = defaultdict(int)
  max_depth_hist = defaultdict(int)
  for i,(probname,(init,deriv,pars,selec,good)) in enumerate(prob_data_list):
    init_len_hist[len(init)] += 1
    sel_len_hist[len(selec)] += 1
    depths = defaultdict(int)
    max_depth = 0
    print(i,probname,len(init),len(deriv),len(pars),len(selec),len(good))
    for id,ps in sorted(pars.items()):
      if not ps: # there
        print(id)
        continue
      
      depth = max([depths[p] for p in ps])+1
      depths[id] = depth
      # print(id,ps,depth)
      if depth>max_depth:
        max_depth = depth
    print("max_depth",max_depth)
    max_depth_hist[max_depth] += 1

  '''
  print("init_len_hist")
  for val,cnt in sorted(init_len_hist.items()):
    print(val,cnt)
  '''
  '''
  print("sel_len_hist")
  for val,cnt in sorted(sel_len_hist.items()):
    print(val,cnt)
  '''
  print("max_depth_hist")
  for val,cnt in sorted(max_depth_hist.items()):
    print(val,cnt)
