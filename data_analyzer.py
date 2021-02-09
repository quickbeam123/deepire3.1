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
  # To be called as in: ./data_analyzer.py training_data.pt

  thax_sign1,sine_sign1,deriv_arits1,thax_to_str1 = torch.load("{}/data_sign.pt".format(sys.argv[1]))

  rule_names = []
  with open("inferences.info.txt","r") as f:
    for line in f:
      rule_names.append(line[:-1])

  for rule in sorted(deriv_arits1):
    print(rule,rule_names[rule] if rule < 666 else "AVATAR")

  exit(0)

  thax_sign2,sine_sign2,deriv_arits2,thax_to_str2 = torch.load("{}/data_sign.pt".format(sys.argv[2]))

  strs2 = {thax_to_str2[thax] for thax in thax_sign2 if thax in thax_to_str2 }
  
  strs2_list = [thax_to_str2[thax] for thax in thax_sign2 if thax in thax_to_str2]

  '''
  for i,thax in enumerate(strs1_list):
    print(thax, i, strs2_list.index(thax))
  '''

  print("len(strs1)=",len(strs1))
  print("len(strs2)=",len(strs2))
  print("len(strs1 & strs2)=",len(strs1 & strs2))
  
  print(len(set(strs2_list[:2000]) - strs1))
  
  exit(0)

  train_data_idx = torch.load("{}/training_index.pt".format(sys.argv[1]))
  print("Loaded train data:",len(train_data_idx))
  valid_data_idx = torch.load("{}/validation_index.pt".format(sys.argv[1]))
  print("Loaded valid data:",len(valid_data_idx))

  data_idx = train_data_idx + valid_data_idx
  
  for size,piece_name in data_idx[-100:]:
    print("{}/pieces/{}".format(sys.argv[1],piece_name))
  
  exit(0)

  prob_data_list = torch.load(sys.argv[1]) # [ probname, (init,deriv,pars,selec,good)]
  
  '''
  for i,(probname,(init,deriv,pars,selec,good)) in enumerate(prob_data_list):
    what = "small_np/" + probname.split("/")[-1][9:-4]
    print(what)
  
  exit(0)
  '''
  
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
  size_hist = defaultdict(int)
  max_depth_hist = defaultdict(int)
  for i,(probname,(init,deriv,pars,selec,good)) in enumerate(prob_data_list):
    init_len_hist[len(init)] += 1
    sel_len_hist[len(selec)] += 1
    size_hist[len(init)+len(deriv)] += 1
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
  '''
  print("max_depth_hist")
  for val,cnt in sorted(max_depth_hist.items()):
    print(val,cnt)
  '''
  print("size_hist")
  tot = 0
  sum = 0
  for val,cnt in sorted(size_hist.items()):
    sum += val*cnt
    tot += cnt
    print(val,cnt)

  print("Average",sum/tot)
