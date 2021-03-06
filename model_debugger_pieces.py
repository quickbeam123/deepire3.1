#!/usr/bin/env python3

import inf_common as IC

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

import numpy as np
import math

from multiprocessing import Pool

import matplotlib.pyplot as plt

# release the memory just claimed - an experiment
import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

def contribute(id,model,pos_vals,neg_vals,posOK,negOK,pos_cuts,neg_cuts,min_pos_logit):  
  # print(id,thax,val,id in selec,id in good)
  logit = model(id) # calling forward
  # print(id,thax,logit,id in selec,id in good)
  val = (logit >= 0.0) # interpreting the logit

  pos = pos_vals[id]
  neg = neg_vals[id]
    
  if pos > 0.0:
    pos_cuts[logit] += pos

    if logit < min_pos_logit:
      min_pos_logit = logit
    
    if val:
      posOK += pos

  if neg > 0.0:
    neg_cuts[logit] += neg

    if not val:
      negOK += neg

  return posOK,negOK,min_pos_logit

def eval_one(task):
  (size,piece_name) = task
  print("Loading",piece_name,"of size",size)

  data = torch.load("{}/pieces/{}".format(sys.argv[1],piece_name))
  (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) = data

  model = torch.jit.load(sys.argv[2]) # always load a new model -- it contains the lookup tables for the particular model

  posOK = 0.0
  negOK = 0.0
  
  # keep collecting the logis values at which we would need to cut to get this clause classified as positive
  pos_cuts = defaultdict(int)
  neg_cuts = defaultdict(int)
  
  min_pos_logit = sys.float_info.max

  for id, (thax,sine) in init:
    if thax == -1:
      st = "-1"
    elif thax in thax_to_str:
      st = thax_to_str[thax]
    else:      
      assert len(thax_to_str) == 0 or thax == 0, thax
      st = str(thax)

    # communication via st and sine
    getattr(model,"new_init")(id,[-1,-1,-1,-1,-1,sine],st)

    posOK,negOK,min_pos_logit = contribute(id,model,pos_vals,neg_vals,posOK,negOK,pos_cuts,neg_cuts,min_pos_logit)

  for id, (rule) in deriv:
    if rule == 666:
      my_pars = pars[id]
      assert(len(my_pars) == 1)
      getattr(model,"new_avat")(id,[-1,-1,-1,my_pars[0]])
    else:
      getattr(model,"new_deriv{}".format(rule))(id,[-1,-1,-1,-1,rule],pars[id])

    posOK,negOK,min_pos_logit = contribute(id,model,pos_vals,neg_vals,posOK,negOK,pos_cuts,neg_cuts,min_pos_logit)

  del model
  libc.malloc_trim(ctypes.c_int(0))

  print("posrate",posOK / tot_pos if tot_pos > 0 else 1.0,"from",posOK,tot_pos)
  print("negrate",negOK / tot_neg if tot_neg > 0 else 1.0,"from",negOK,tot_neg)
  print("tot weight",tot_pos+tot_neg)

  return (piece_name,posOK,tot_pos,negOK,tot_neg,pos_cuts,neg_cuts,min_pos_logit)

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Load a torchscript model and folder with piece indexes
  # test the model on the pieces and report individual and average pos/neg rates
  # finally, plot a logits graph
  #
  # To be called as in: ./model_debugger_pieces.py <folder1> torch_script_model.pt TRAIN/VALID/<nothing>
  # 
  # <folder1> to contain "training_data.pt" and "validation_data.pt"

  # thax_sign,sine_sign,deriv_arits,thax_to_str 
  # thax_sign,sine_sign,deriv_arits,thax_to_str,prob_name2id,prob_id2name (added in gsd)
  signature = torch.load("{}/data_sign.pt".format(sys.argv[1]))
  thax_sign = signature[0]
  sine_sign = signature[1]
  deriv_arits = signature[2]
  thax_to_str = signature[3]
  
  print("Loaded data signature")  
  train_data_idx = torch.load("{}/training_index.pt".format(sys.argv[1]))
  print("Loaded train data:",len(train_data_idx))
  valid_data_idx = torch.load("{}/validation_index.pt".format(sys.argv[1]))
  print("Loaded valid data:",len(valid_data_idx))
  
  specifier = sys.argv[3] if len(sys.argv)>3 else "ALL"
  if specifier=="TRAIN":
    data_idx = train_data_idx
  elif specifier=="VALID":
    data_idx = valid_data_idx
  else:
    data_idx = train_data_idx + valid_data_idx
  
  '''
  results = []
  for task in data_idx[:3]:
    res = eval_one(task)
    print("Done",task[0])
    results.append(res)
  '''
  pool = Pool(processes=70)
  results = pool.map(eval_one, data_idx, chunksize = 5)
  pool.close()
  pool.join()
  del pool

  cnt = 0
  posrate_sum = 0.0
  negrate_sum = 0.0
  
  posOK_sum = 0.0
  posTot_sum = 0.0
  negOK_sum = 0.0
  negTot_sum = 0.0
  
  per_prob = defaultdict(list)
  pos_cuts_fin = defaultdict(int)
  neg_cuts_fin = defaultdict(int)
  
  for (probname,posOK,posTot,negOK,negTot,pos_cuts,neg_cuts,min_pos_logit) in results:
    
    posOK_sum += posOK
    negOK_sum += negOK
    posTot_sum += posTot
    negTot_sum += negTot
    
    posrate = posOK / posTot if posTot > 0 else 1.0
    negrate = negOK / negTot if negTot > 0 else 1.0
    
    cnt += 1
    print(probname)
    print(cnt,"Posrate",posrate,"negrate",negrate)
  
    posrate_sum += posrate
    negrate_sum += negrate

    for logit, val in pos_cuts.items():
      pos_cuts_fin[logit] += val
    
    for logit, val in neg_cuts.items():
      neg_cuts_fin[logit] += val

    per_prob[min_pos_logit].append(probname)

  print()
  print("Total probs:",cnt)
  print("Final (average) posrate:",posrate_sum/cnt)
  print("Final (average) negrate:",negrate_sum/cnt)

  print()
  print("Aggreg posRate",posOK_sum/posTot_sum)
  print("Aggreg negRate",negOK_sum/negTot_sum)

  print()
  print("per_prob pos example logit minima")
  for logit, problems in sorted(per_prob.items()):
    print(logit,len(problems))

  print()
  print("Plotting")

  logits = []
  cur_pos = 0
  pos_vals = []
  cur_neg = 0
  neg_vals = []
  cur_prob = 0
  
  # for the ROC curve (thanks to Filip)
  logit_threshold_vals = []
  logit_threshold_idxs = []
  
  prob_logits = []
  prob_vals = []
  last_logit = None
  for logit in sorted(set(pos_cuts_fin) | set(neg_cuts_fin)):
    if last_logit:
      c = math.ceil(last_logit)
      if logit > c:
        logit_threshold_vals.append(c)
        logit_threshold_idxs.append(len(logits))
    last_logit = logit
  
    logits.append(logit)
    cur_pos -= pos_cuts_fin[logit]
    pos_vals.append(cur_pos)
    cur_neg += neg_cuts_fin[logit]
    neg_vals.append(cur_neg)
    
    if per_prob[logit]:
      prob_logits.append(logit)
      cur_prob += len(per_prob[logit])
      prob_vals.append(cur_prob)

  fig, ax1 = plt.subplots(figsize=(4, 3))

  color = 'tab:blue'
  ax1.set_xlabel('logit value threshold')
  ax1.set_ylabel('clause volume', color=color)
  lnv, = ax1.plot(logits, np.array(neg_vals)/neg_vals[-1], ".", markersize = 2, label = "TNR", color="green")
  lps, = ax1.plot(logits, (np.array(pos_vals)-pos_vals[-1])/(-pos_vals[-1]), ".", markersize = 2,label = "TPR", color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax1.axhline(0.0,linewidth=1)
  ax1.axhline(1.0,linewidth=1)

  ax1.axvline(0.0,linewidth=1)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:red'
  ax2.set_ylabel('problem volume', color=color)  # we already handled the x-label with ax1

  lpv, = ax2.plot(prob_logits, np.array(prob_vals)/prob_vals[-1], "+", linewidth = 1, label = "min", color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  low1,high1 = ax1.get_ylim()
  low2,high2 = ax2.get_ylim()

  # print(low1,high1)
  # print(low2,high2)
  # ax2.set_ylim((-0.5,high2))

  fig.tight_layout()  # otherwise the right y-label is slightly clipped

  plt.legend(handles = [lps,lnv,lpv], loc='upper left', handletextpad=0.1) # loc = 'best' is rumored to be unpredictable

  # plt.show()

  filename = "debugger_plot_{}_{}.png".format(sys.argv[2].split("/")[-1],specifier)
  plt.savefig(filename,dpi=250)
  print("Saved final plot to",filename)
  plt.close(fig)

  # the ROC curve - thanks to Filip. Actually, in my curve, 0 and 1 and flipped, I think
  fig, ax = plt.subplots(figsize=(3, 3))
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')

  Xs = 1.0 - np.array(neg_vals)/neg_vals[-1]
  Ys = 1.0 - np.array(pos_vals)/pos_vals[-1]

  line, = ax.plot([0.0,1.0],[0.0,1.0], "-", linewidth = 1, color = "gray")
  line, = ax.plot(Xs,Ys, "-")

  for val,idx in zip(logit_threshold_vals,logit_threshold_idxs):
    ax.annotate(str(val),xy=(Xs[idx],Ys[idx]))

  fig.tight_layout()

  filename = "ROC_curve_{}_{}.png".format(sys.argv[2].split("/")[-1],specifier)
  plt.savefig(filename,dpi=250)
  print("Saved ROC curve to",filename)
  plt.close(fig)
