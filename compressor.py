#!/usr/bin/env python3

import inf_common as IC

import torch
from torch import Tensor

import time,bisect,random

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
  # To be called as in: ./data_analyzer.py data_hist.pt training_data.pt <compressed_data_file_name>

  init_hist,deriv_hist = torch.load(sys.argv[1])
  parts = IC.get_initial_model(init_hist,deriv_hist)

  prob_data_list = torch.load(sys.argv[2]) # [ probname, (init,deriv,pars,selec,good)]
  
  size_hist = defaultdict(int)
  
  sizes = []
  times = []
  
  size_and_prob = []
  
  for i,(probname,(init,deriv,pars,selec,good)) in enumerate(prob_data_list):
    size = len(init)+len(deriv)
    
    size_and_prob.append((size,(probname,(init,deriv,pars,selec,good))))
    
    size_hist[len(init)+len(deriv)] += 1
    # print(i,probname,size)
    '''
    start_time = time.time()
    model = IC.LearningModel(*parts,init,deriv,pars,selec,good)
    (loss,posRate,negRate) = model()
    model_time = time.time()-start_time
    print("Model in",model_time)
  
    sizes.append(size)
    times.append(model_time)
    '''

  '''
  import matplotlib.pyplot as plt
  plt.scatter(sizes, times)
  plt.savefig("times_scatter.png",dpi=250)
  '''

  print("size_hist")
  tot = 0
  sum = 0
  small = 0
  big = 0
  for val,cnt in sorted(size_hist.items()):
    sum += val*cnt
    tot += cnt
    # print(val,cnt)
    if val > 10000:
      big += cnt
    else:
      small += cnt
  print("Average",sum/tot)
  print("Big",big)
  print("Small",small)

  treshold = 10000
  print("Compressing for treshold",treshold)
  size_and_prob.sort()
  
  compressed = []
  
  while size_and_prob:
    size, my_rest = size_and_prob.pop()

    print("Poped guy of size",size)

    while size < treshold and size_and_prob:
      print("Looking for a friend")
      likes_sizes = int((treshold-size)*1.2)
      idx_upper = bisect.bisect_right(size_and_prob,(likes_sizes, my_rest))

      if not idx_upper:
        idx_upper = 1

      idx = random.randrange(idx_upper)
    
      print("Idxupper",idx_upper,"idx",idx)

      friend_size, friend_rest = size_and_prob[idx]
      del size_and_prob[idx]

      print("friend_size",friend_size)

      my_rest = IC.compress_prob_data([my_rest,friend_rest])
      probname, (init,deriv,pars,selec,good) = my_rest
      size = len(init)+len(deriv)
    
      print("aftermerge",size)

    print("Storing a guy of size",size)
    compressed.append(my_rest)

  print()
  print("Compressed to",len(compressed),"merged problems")
  torch.save(compressed,sys.argv[3])
  print("Save to",sys.argv[3])
