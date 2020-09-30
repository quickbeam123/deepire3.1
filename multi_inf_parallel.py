#!/usr/bin/env python3

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

import numpy as np

import inf_common as IC

NUMPROCESSES = 50

def copy_parts_and_zero_grad_in_copy(parts,parts_copies):
  for part,part_copy in zip(parts,parts_copies):
    part_copy.load_state_dict(part.state_dict())

  for param in parts_copies.parameters():
    # taken from Optmizier zero_grad, roughly
    if param.grad is not None:
      param.grad.detach_()
      param.grad.zero_()

def copy_grads_back_from_param(parts,parts_copies):
  for param, param_copy in zip(parts.parameters(),parts_copies.parameters()):
    # print("Copy",param_copy)
    # print("Copy.grad",param_copy.grad)
    param.grad = param_copy

def learn_on_one(myparts,data):

  # TODO: removeme - doing "deep copy via pickling"
  # torch.save(myparts,"tmp")
  # myparts = torch.load("tmp")

  # probname = prob_data_list[idx][0]
  # data = prob_data_list[idx][1]
  (init,deriv,pars,selec,good) = data
  model = IC.LearningModel(*myparts,init,deriv,pars,selec,good)
  (loss,posRate,negRate) = model()
  loss.backward()
  
  # put grad into actual tensor to be returned below (gradients don't go through the Queue)
  for param in myparts.parameters():
    grad = param.grad
    # print("Before",param)
    # print("Its grad",grad)
    param.requires_grad = False # to allow the in place operation just below
    if grad is not None:
      param.copy_(grad)
    else:
      param.zero_()
    param.requires_grad = True # to be ready for the next learning when assigned to a new job
    # print("After",param)
  
  return (loss[0].item(),posRate,negRate,myparts)

def worker(q_in, q_out):
  while True:
    # print("Get by",os.getpid())
    (idx,myparts,data) = q_in.get()
    (loss,posRate,negRate,myparts) = learn_on_one(myparts,data)
    q_out.put((idx,loss,posRate,negRate,myparts))

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Learn in parallel using a Pool of processes, or something similar
  #
  # probably needs to be run with "ulimit -Sn 3000" or something large
  # To be called as in: ./multi_inf_parallel.py enigma_smt_447/training_data.pt enigma_smt_447/model0_4_Tanh.pt where_to_save_models_prefix

  prob_data_list = torch.load(sys.argv[1])
  print("Loaded",sys.argv[1])

  master_parts = torch.load(sys.argv[2])
  parts_copies = [] # have as many copies as processes; they are kind of shared by Queue, so only one process should touch one at a time
  for i in range(NUMPROCESSES):
    parts_copies.append(torch.load(sys.argv[2])) # currently, don't know how to reliably deep-copy in memory (with pickling, all seems fine)
  
  print("Loaded",sys.argv[2])

  q_in = torch.multiprocessing.Queue()
  q_out = torch.multiprocessing.Queue()
  for i in range(NUMPROCESSES):
    p = torch.multiprocessing.Process(target=worker, args=(q_in,q_out))
    p.start()
  
  t = 0
  
  statistics = np.tile([1.0,0.0,0.0],(len(prob_data_list),1)) # the last recoreded stats on the i-th problem
  
  # init learning
  best_loss = 1000.0
  
  optimizer = torch.optim.Adam(master_parts.parameters(), lr=IC.LEARN_RATE)

  feed_sequence = []

  while True:
    while parts_copies:
      parts_copy = parts_copies.pop()
      
      if not feed_sequence:
        feed_sequence = list(range(len(prob_data_list)))
        random.shuffle(feed_sequence)
      idx = feed_sequence.pop()
        
      (probname,data) = prob_data_list[idx]
      copy_parts_and_zero_grad_in_copy(master_parts,parts_copy)
      t += 1
        
      print("Time",t,"starting job on problem",idx,probname,"size",len(data[-2]))
      print()
      
      q_in.put((idx,parts_copy,data))
      
    # time.sleep(30.0)
    
    (idx,loss,posRate,negRate,his_parts) = q_out.get() # this may block
    parts_copies.append(his_parts)
    
    copy_grads_back_from_param(master_parts,his_parts)
    optimizer.step()

    print("Job finished at on problem",idx)
    print("Local:",loss,posRate,negRate)
    statistics[idx] = (loss,posRate,negRate)

    (loss,posRate,negRate) = np.mean(statistics,axis=0)
    print("Global:",loss,posRate,negRate,flush=True)
    print()

    if (t % len(prob_data_list) == 0):
      name = sys.argv[3]+"/models/periody_{}_{}_{}_l{}_p{}_n{}.pt".format(t//len(prob_data_list),IC.EMBED_SIZE,str(IC.NONLIN)[:4],loss,posRate,negRate)
      print("Period reached, saving to:",name)
      print()
      torch.save(master_parts,name)

    if t>len(prob_data_list) and (loss < best_loss):
      name = sys.argv[3]+"/models/improvy_{}_{}_{}_l{}_p{}_n{}.pt".format(t,IC.EMBED_SIZE,str(IC.NONLIN)[:4],loss,posRate,negRate)
      
      print("Improved best, saving to:",name)
      print()
      
      torch.save(master_parts,name)

      best_loss = loss

