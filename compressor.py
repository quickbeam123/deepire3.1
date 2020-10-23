#!/usr/bin/env python3

import inf_common as IC

import torch
from torch import Tensor

import time,bisect,random,math,os,errno

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

def compress_to_treshold(prob_data_list,treshold):
  
  size_hist = defaultdict(int)
  
  sizes = []
  times = []
  
  size_and_prob = []
  
  for i,(probname,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg)) in enumerate(prob_data_list):
    size = len(init)+len(deriv)
    
    size_and_prob.append((size,(probname,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg))))
    
    size_hist[len(init)+len(deriv)] += 1

  print("size_hist")
  tot = 0
  sum = 0
  small = 0
  big = 0
  for val,cnt in sorted(size_hist.items()):
    sum += val*cnt
    tot += cnt
    # print(val,cnt)
    if val > treshold:
      big += cnt
    else:
      small += cnt
  print("Average",sum/tot)
  print("Big",big)
  print("Small",small)

  print("Compressing for treshold",treshold)
  size_and_prob.sort()
  
  compressed = []
  
  while size_and_prob:
    size, my_rest = size_and_prob.pop()

    # print("Popped guy of size",size)

    while size < treshold and size_and_prob:
      # print("Looking for a friend")
      likes_sizes = int((treshold-size)*1.2)
      idx_upper = bisect.bisect_right(size_and_prob,(likes_sizes, my_rest))

      if not idx_upper:
        idx_upper = 1

      idx = random.randrange(idx_upper)
    
      # print("Idxupper",idx_upper,"idx",idx)

      friend_size, friend_rest = size_and_prob[idx]
      del size_and_prob[idx]

      # print("friend_size",friend_size)

      my_rest = IC.compress_prob_data([my_rest,friend_rest])
      probname, (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) = my_rest
      size = len(init)+len(deriv)
    
      # print("aftermerge",size)

    # print("Storing a guy of size",size)
    compressed.append(my_rest)

  print()
  print("Compressed to",len(compressed),"merged problems")
  return compressed

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # To be called as in: ./compressor.py <folder> raw_log_data_*.pt
  #
  # raw_log_data is compressed via abstraction (and a smoothed representation is created)
  #
  # optionally, multiple problems can be grouped together (also using the compression code)
  #
  # finally, 80-20 split on the suffled list is performed and training_data.pt validation_data.pt are saved to folder

  prob_data_list = torch.load(sys.argv[2])
  
  # prob_data_list = prob_data_list[:10]
  
  print("Loaded raw prob_data_list of len:",len(prob_data_list))

  print("Dropping axiom information, not needed anymore")
  prob_data_list = [(probname,(init,deriv,pars,selec,good)) for (probname,(init,deriv,pars,selec,good,axioms)) in prob_data_list]
  print("Done")

  print("Smoothed representation")
  for i, (probname,(init,deriv,pars,selec,good)) in enumerate(prob_data_list):
    pos_vals = defaultdict(float)
    neg_vals = defaultdict(float)
    tot_pos = 0.0
    tot_neg = 0.0

    one_clause_weigth = 1.0/len(selec)

    for id in selec:
      if id in good:
        pos_vals[id] = one_clause_weigth
        tot_pos += one_clause_weigth
      else:
        neg_vals[id] = one_clause_weigth
        tot_neg += one_clause_weigth

    prob_data_list[i] = (probname,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg))
    
    '''
    if "t16_finsub_1" in probname:
      print(probname)
      print(selec)
      print(good)
      print(pos_vals)
      print(neg_vals)
    '''
    
  print("Done")

  print("Compressing")
  for i, (probname,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg)) in enumerate(prob_data_list):
    print(probname,"init: {}, deriv: {}, pos_vals: {}, neg_vals: {}".format(len(init),len(deriv),len(pos_vals),len(neg_vals)))
    prob_data_list[i] = IC.compress_prob_data([(probname,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg))])
  print("Done")

  print("Making smooth compression discreet again")
  for i, (probname,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg)) in enumerate(prob_data_list):
    print()
  
    print(probname)
    print(tot_pos,tot_neg)
  
    '''
    if "t16_finsub_1" in probname:
      print(pos_vals)
      print(neg_vals)
      print(tot_pos)
      print(tot_neg)
    '''
  
    tot_pos = 0.0
    tot_neg = 0.0
            
    for id,val in neg_vals.items():
      if id in pos_vals and pos_vals[id] > 0.0: # pos has priority
        '''
        if val != 1.0:
          print("negval goes from",val,"to 0.0 for posval",pos_vals[id])
        '''
        neg_vals[id] = 0.0
      elif val > 0.0:
        '''
        if val != 1.0:
          print("negval goes from",val,"to 1.0")
        '''
        neg_vals[id] = 1.0 # neg counts as one
        tot_neg += 1.0

    for id,val in pos_vals.items():
      if val > 0.0:
        '''
        if val != 1.0:
          print("posval goes from",val,"to 1.0")
        '''
        pos_vals[id] = 1.0 # pos counts as one too
        tot_pos += 1.0

    # new stuff -- normalize so that each abstracted clause in a problem has so much "voice" that tha whole problem has a sum of 1.0
    factor = 1.0/(tot_pos+tot_neg)
    for id,val in pos_vals.items():
      pos_vals[id] *= factor
    for id,val in neg_vals.items():
      neg_vals[id] *= factor
    tot_pos *= factor
    tot_neg *= factor

    '''
    if "t16_finsub_1" in probname:
      print(pos_vals)
      print(neg_vals)
      print(tot_pos)
      print(tot_neg)
    '''

    print(tot_pos,tot_neg)

    prob_data_list[i] = (probname,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg))

  if False: # Big compression now:
    print("Grand compression")
    (joint_probname,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg)) = IC.compress_prob_data(prob_data_list)

    '''
    for id in sorted(set(pos_vals) | set(neg_vals)):
      print(id, pos_vals[id], neg_vals[id])
    print(tot_pos)
    print(tot_neg)
    '''

    filename = "{}/big_blob.pt".format(sys.argv[1])
    print("Saving big blob to",filename)
    torch.save((init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg), filename)

    print("Done")

  if True:
    prob_data_list = compress_to_treshold(prob_data_list,treshold = 5000)

  print("Saving pieces")
  dir = "{}/pieces".format(sys.argv[1])
  try:
    os.mkdir(dir)
  except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass
  for i,(probname,rest) in enumerate(prob_data_list):
    piece_name = "piece{}.pt".format(i)
    torch.save(rest, "{}/{}".format(dir,piece_name))
    prob_data_list[i] = (len(rest[0])+len(rest[1]),piece_name)
  print("Done")

  random.shuffle(prob_data_list)
  spl = math.ceil(len(prob_data_list) * 0.8)
  print("shuffled and split at idx",spl,"out of",len(prob_data_list))
  print()

  # save just names:
  filename = "{}/training_index.pt".format(sys.argv[1])
  print("Saving training part to",filename)
  torch.save(prob_data_list[:spl], filename)
  filename = "{}/validation_index.pt".format(sys.argv[1])
  print("Saving validation part to",filename)
  torch.save(prob_data_list[spl:], filename)

  exit(0)

  # the old way below:

  filename = "{}/training_data.pt".format(sys.argv[1])
  print("Saving training part to",filename)
  torch.save(prob_data_list[:spl], filename)
  filename = "{}/validation_data.pt".format(sys.argv[1])
  print("Saving validation part to",filename)
  torch.save(prob_data_list[spl:], filename)
