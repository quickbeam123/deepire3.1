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
import hyperparams as HP

NUMPROCESSES = 20

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

def eval_and_or_learn_on_one(myparts,data,training):
  # probname = prob_data_list[idx][0]
  # data = prob_data_list[idx][1]
  (init,deriv,pars,selec,good) = data
  model = IC.LearningModel(*myparts,init,deriv,pars,selec,good)
  
  if training:
    model.train()
  else:
    model.eval()
  
  (loss,posRate,negRate) = model()
  
  if training:
    loss.backward()
    # put grad into actual tensor to be returned below (gradients don't go through the Queue)
    for param in myparts.parameters():
      grad = param.grad
      param.requires_grad = False # to allow the in-place operation just below
      if grad is not None:
        param.copy_(grad)
      else:
        param.zero_()
      param.requires_grad = True # to be ready for the next learning when assigned to a new job
  
  return (loss[0].item(),posRate,negRate,myparts)

def worker(q_in, q_out):
  while True:
    (idx,myparts,data,training) = q_in.get()
    print("Child",os.getpid(),"has",idx)
    (loss,posRate,negRate,myparts) = eval_and_or_learn_on_one(myparts,data,training)
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
  #
  # To be called as in: ./multi_inf_parallel.py <folder> <initial-model>
  #
  # it expects <folder> to contain "training_data.pt" and "validation_data.pt"
  # (and maybe also "data_hist.pt")
  # if <initial-model> is not specified,
  # it creates a new one in <folder> using the same naming scheme as initializer.py
  
  train_data_list = torch.load("{}/training_data.pt".format(sys.argv[1]))
  print("Loaded train data:",len(train_data_list))
  valid_data_list = torch.load("{}/validation_data.pt".format(sys.argv[1]))
  print("Loaded valid data:",len(valid_data_list))
  
  if len(sys.argv) >= 3:
    master_parts = torch.load(sys.argv[2])
    print("Loaded model parts",sys.argv[2])
  else:
    init_hist,deriv_hist = torch.load("{}/data_hist.pt".format(sys.argv[1]))
    master_parts = IC.get_initial_model(init_hist,deriv_hist)
    model_name = "{}/initial{}".format(sys.argv[1],IC.name_initial_model_suffix())
    torch.save(master_parts,model_name)
    print("Created model parts and saved to",model_name)

  epoch = 0

  # in addition to the "oficial model" as named above, we checkpoint it as epoch0 here.
  model_name = "{}/model-epoch{}.pt".format(sys.argv[1],epoch)
  torch.save(master_parts,model_name)

  parts_copies = [] # have as many copies as processes; they are somehow shared among the processes via Queue, so only one process should touch one at a time
  for i in range(NUMPROCESSES):
    parts_copies.append(torch.load(model_name)) # currently, don't know how to reliably deep-copy in memory (with pickling, all seems fine)

  q_in = torch.multiprocessing.Queue()
  q_out = torch.multiprocessing.Queue()
  for i in range(NUMPROCESSES):
    p = torch.multiprocessing.Process(target=worker, args=(q_in,q_out))
    p.start()
  
  t = 0

  # this is never completely faithful, since the model updates continually
  train_statistics = np.tile([1.0,0.0,0.0],(len(train_data_list),1)) # the last recoreded stats on the i-th problem
  validation_statistics = np.tile([1.0,0.0,0.0],(len(valid_data_list),1))

  optimizer = torch.optim.Adam(master_parts.parameters(), lr=HP.LEARN_RATE)

  times = []
  train_losses = []
  train_posrates = []
  train_negrates = []
  valid_losses = []
  valid_posrates = []
  valid_negrates = []
  
  start_time = time.time()

  EPOCHS_BEFORE_VALIDATION = 1

  while True:
    epoch += EPOCHS_BEFORE_VALIDATION
    
    if epoch > 50:
      exit(0)
    
    times.append(epoch)
    
    feed_sequence = []
    for _ in range(EPOCHS_BEFORE_VALIDATION):
      # SGDing - so traverse each time in new order
      epoch_bit = list(range(len(train_data_list)))
      random.shuffle(epoch_bit)
      feed_sequence += epoch_bit
  
    # training on each problem in these EPOCHS_BEFORE_VALIDATION-many epochs
    while feed_sequence or len(parts_copies) < NUMPROCESSES:
      # we use parts_copies as a counter of idle children in the pool
      while parts_copies and feed_sequence:
        parts_copy = parts_copies.pop()
        idx = feed_sequence.pop()
        
        (probname,data) = train_data_list[idx]
        copy_parts_and_zero_grad_in_copy(master_parts,parts_copy)
        t += 1
        print("Time",t,"starting training job on problem",idx,probname,"size",len(data[-2]))
        print()
        q_in.put((idx,parts_copy,data,True)) # True stands for "training is on"

      (idx,loss,posRate,negRate,his_parts) = q_out.get() # this may block
      parts_copies.append(his_parts) # increase the ``counter'' again

      copy_grads_back_from_param(master_parts,his_parts)
      optimizer.step()

      print("Job finished at on problem",idx,flush=True)
      print("Local:",loss,posRate,negRate)
      train_statistics[idx] = (loss,posRate,negRate)

    print()
    print("(Multi)-epoch",epoch,"learning finished at",time.time() - start_time)
    model_name = "{}/model-epoch{}.pt".format(sys.argv[1],epoch)
    print("Saving model to:",model_name)
    torch.save(master_parts,model_name)

    (loss,posRate,negRate) = np.mean(train_statistics,axis=0)
    print("Training stats:",loss,posRate,negRate,flush=True)
    print("Validating...")
    
    train_losses.append(loss)
    train_posrates.append(posRate)
    train_negrates.append(negRate)

    feed_sequence = list(range(len(valid_data_list)))
    while feed_sequence or len(parts_copies) < NUMPROCESSES:
      # we use parts_copies as a counter of idle children in the pool
      while parts_copies and feed_sequence:
        parts_copy = parts_copies.pop()
        idx = feed_sequence.pop()
        
        (probname,data) = valid_data_list[idx]
        copy_parts_and_zero_grad_in_copy(master_parts,parts_copy)
        t += 1
        print("Time",t,"starting validation job on problem",idx,probname,"size",len(data[-2]))
        print()
        q_in.put((idx,parts_copy,data,False)) # False stands for "training is off"

      (idx,loss,posRate,negRate,his_parts) = q_out.get() # this may block
      parts_copies.append(his_parts) # increase the ``counter'' again

      print("Job finished at on problem",idx,flush=True)
      print("Local:",loss,posRate,negRate)
      validation_statistics[idx] = (loss,posRate,negRate)

    (loss,posRate,negRate) = np.mean(validation_statistics,axis=0)
    print("(Multi)-epoch",epoch,"validation finished at",time.time() - start_time)
    print("Validation stats:",loss,posRate,negRate,flush=True)

    valid_losses.append(loss)
    valid_posrates.append(posRate)
    valid_negrates.append(negRate)

    # plotting
    IC.plot_one("plot.png",times,train_losses,train_posrates,train_negrates,valid_losses,valid_posrates,valid_negrates)




