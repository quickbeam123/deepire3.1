#!/usr/bin/env python3

import inf_common as IC

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

def eval_one(model,init,deriv,pars,selec,good,axioms):
  posOK = 0
  posTot = 0
  negOK = 0
  negTot = 0

  for id, thax in init:
    if thax == -1:
      st = "-1"
    elif id in axioms:
      st = axioms[id]
    else:
      st = str(thax)
    getattr(model,"new_init")(id,[0,0,0,1,0,0],st)

    # print(id,thax,val,id in selec,id in good)
    logit = model(id) # calling forward
    # print(id,thax,logit,id in selec,id in good)
    val = (logit >= 0.0) # interpreting the logit

    if id in selec:
      if id in good:
        posTot += 1
        if val:
          posOK += 1
      else:
        negTot += 1
        if not val:
          negOK += 1

  for id, rule in deriv:
    if rule == 666:
      my_pars = pars[id]
      assert(len(my_pars) == 1)
      getattr(model,"new_avat")(id,[0,0,0,my_pars[0]])
    else:
      getattr(model,"new_deriv{}".format(rule))(id,[0,0,0,0,rule],pars[id])

    logit = model(id) # calling forward
    # print(id,rule,logit,id in selec,id in good)
    val = (logit >= 0.0) # interpreting the logit

    if id in selec:
      if id in good:
        posTot += 1
        if val:
          posOK += 1
      else:
        negTot += 1
        if not val:
          negOK += 1

  return (posOK,posTot,negOK,negTot)

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Load a torchscript model and a set of logs, passed in a file as the final argument,
  # test the model on the logs (as if vampire was running) and report individual and average pos/neg rates
  #
  # To be called as in: ./model_debugger.py torch_script_model.pt *.log-files-listed-line-by-line-in-a-file (an "-s4k on" run of vampire)

  # loading logs as in log_loader

  cnt = 0
  posrate_sum = 0.0
  negrate_sum = 0.0

  seen = 0

  with open(sys.argv[2],"r") as f:
    for line in f:
      probname = line[:-1]
      probdata = IC.load_one(probname)
      if probdata is None:
        continue
    
      (init,deriv,pars,selec,good,axioms) = probdata

      model = torch.jit.load(sys.argv[1]) # always load a new model -- it contains the lookup tables for the particular model

      (posOK,posTot,negOK,negTot) = eval_one(model,init,deriv,pars,selec,good,axioms)
      
      posrate = posOK / posTot if posTot > 0 else 1.0
      negrate = negOK / negTot if negTot > 0 else 1.0
      
      cnt += 1
      print(cnt,"Posrate",posrate,"negrate",negrate)
      posrate_sum += posrate
      negrate_sum += negrate

      seen += 1
      if seen >= 1000:
        break

  print()
  print("Total probs:",cnt)
  print("Final posrate:",posrate_sum/cnt)
  print("Final negrate:",negrate_sum/cnt)


