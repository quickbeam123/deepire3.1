#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

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

import os

NUMPROCESSES = 5

DATA_THROUGH_QUEUE = True

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
  (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) = data
  
  # print("Datum of size",len(init)+len(deriv))
  
  model = IC.LearningModel(*myparts,init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg)
  
  # print("Model created")
  
  if training:
    model.train()
  else:
    model.eval()
  
  (loss,posRate,negRate) = model()
  
  # print("Model evaluated")
  
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

    # print("Training finished")

  return (loss[0].item(),posRate,negRate,myparts)

global common_data
common_data = None

def worker(q_in, q_out):
  global common_data
  while True:
    if DATA_THROUGH_QUEUE:
      (idx,data,myparts,training) = q_in.get()
    else:
      (idx,myparts,training) = q_in.get()
      probname, data = common_data[training][idx]
    
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
  # To be called as in: ./multi_inf_parallel.py <folder_in> <folder_out> <initial-model>
  #
  # it expects <folder_in> to contain "training_data.pt" and "validation_data.pt"
  # (and maybe also "data_sign.pt")
  #
  # if <initial-model> is not specified,
  # it creates a new one in <folder_out> using the same naming scheme as initializer.py
  #
  # The log, the plot, and intermediate models are also saved in <folder_out>
  
  # global redirect of prints to the just upen "logfile"
  sys.stdout = open("{}/run{}".format(sys.argv[2],IC.name_learning_regime_suffix()), 'w')
  
  train_data_list = torch.load("{}/training_data.pt".format(sys.argv[1]))
  print("Loaded train data:",len(train_data_list))
  valid_data_list = torch.load("{}/validation_data.pt".format(sys.argv[1]))
  print("Loaded valid data:",len(valid_data_list))
  
  if len(sys.argv) >= 4:
    master_parts = torch.load(sys.argv[3])
    print("Loaded model parts",sys.argv[3])
  else:
    thax_sign,sine_sign,deriv_arits,thax_to_str = torch.load("{}/data_sign.pt".format(sys.argv[1]))
    master_parts = IC.get_initial_model(thax_sign,sine_sign,deriv_arits)
    model_name = "{}/initial{}".format(sys.argv[2],IC.name_initial_model_suffix())
    torch.save(master_parts,model_name)
    print("Created model parts and saved to",model_name)

  if HP.TRR == HP.TestRiskRegimen_OVERFIT:
    # merge validation data back to training set (and ditch early stopping regularization)
    train_data_list += valid_data_list
    valid_data_list = []
    print("Merged valid with train; final:",len(train_data_list))

  common_data = [valid_data_list,train_data_list]

  epoch = 0

  # in addition to the "oficial model" as named above, we checkpoint it as epoch0 here.
  model_name = "{}/model-epoch{}.pt".format(sys.argv[2],epoch)
  torch.save(master_parts,model_name)

  NUM_ACTIVE_TASKS = NUMPROCESSES

  parts_copies = [] # have as many copies as NUM_ACTIVE_TASKS; they are somehow shared among the processes via Queue, so only one process should touch one at a time
  for i in range(NUM_ACTIVE_TASKS):
    parts_copies.append(torch.load(model_name)) # currently, don't know how to reliably deep-copy in memory (with pickling, all seems fine)

  q_in = torch.multiprocessing.Queue()
  q_out = torch.multiprocessing.Queue()
  my_processes = []
  for i in range(NUMPROCESSES):
    p = torch.multiprocessing.Process(target=worker, args=(q_in,q_out))
    p.start()
    my_processes.append(p)
  
  t = 0

  # this is never completely faithful, since the model updates continually
  train_statistics = np.tile([1.0,0.0,0.0],(len(train_data_list),1)) # the last recoreded stats on the i-th problem
  validation_statistics = np.tile([1.0,0.0,0.0],(len(valid_data_list),1))

  if HP.OPTIMIZER == HP.Optimizer_SGD: # could also play with momentum and its dampening here!
    optimizer = torch.optim.SGD(master_parts.parameters(), lr=HP.LEARN_RATE)
  elif HP.OPTIMIZER == HP.Optimizer_ADAM:
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
   
    if epoch > 100:
      break
    
    times.append(epoch)

    feed_sequence = []
    for _ in range(EPOCHS_BEFORE_VALIDATION):
      # SGDing - so traverse each time in new order
      epoch_bit = list(range(len(train_data_list)))
      random.shuffle(epoch_bit)
      feed_sequence += epoch_bit
  
    # training on each problem in these EPOCHS_BEFORE_VALIDATION-many epochs
    while feed_sequence or len(parts_copies) < NUM_ACTIVE_TASKS:
      # we use parts_copies as a counter of idle children in the pool
      while parts_copies and feed_sequence:
        parts_copy = parts_copies.pop()
        idx = feed_sequence.pop()
        
        (probname,data) = train_data_list[idx]
        copy_parts_and_zero_grad_in_copy(master_parts,parts_copy)
        t += 1
        print("Time",t,"starting training job on problem",idx,probname,"size",len(data[0])+len(data[1]))
        print()
        if DATA_THROUGH_QUEUE:
          message = (idx,data,parts_copy,True) # True stands for "training is on"
        else:
          message = (idx,parts_copy,True) # True stands for "training is on"
        q_in.put(message)

      (idx,loss,posRate,negRate,his_parts) = q_out.get() # this may block
      parts_copies.append(his_parts) # increase the ``counter'' again

      copy_grads_back_from_param(master_parts,his_parts)
      optimizer.step()

      print("Job finished at on problem",idx,flush=True)
      print("Local:",loss,posRate,negRate)
      train_statistics[idx] = (loss,posRate,negRate)

    print()
    print("(Multi)-epoch",epoch,"learning finished at",time.time() - start_time)
    model_name = "{}/model-epoch{}.pt".format(sys.argv[2],epoch)
    print("Saving model to:",model_name)
    torch.save(master_parts,model_name)

    (loss,posRate,negRate) = np.mean(train_statistics,axis=0)
    loss *= len(train_statistics) # we want the loss summed up; so that mergeing preserves comparable values
    print("Training stats:",loss,posRate,negRate,flush=True)
    print("Validating...")
    
    train_losses.append(loss)
    train_posrates.append(posRate)
    train_negrates.append(negRate)

    feed_sequence = list(range(len(valid_data_list)))
    while feed_sequence or len(parts_copies) < NUM_ACTIVE_TASKS:
      # we use parts_copies as a counter of idle children in the pool
      while parts_copies and feed_sequence:
        parts_copy = parts_copies.pop()
        idx = feed_sequence.pop()
        
        (probname,data) = valid_data_list[idx]
        copy_parts_and_zero_grad_in_copy(master_parts,parts_copy)
        t += 1
        print("Time",t,"starting validation job on problem",idx,probname,"size",len(data[0])+len(data[1]))
        print()
        if DATA_THROUGH_QUEUE:
          message = (idx,data,parts_copy,False) # False stands for "training is off"
        else:
          message = (idx,parts_copy,False) # False stands for "training is off"
        
        q_in.put(message)

      (idx,loss,posRate,negRate,his_parts) = q_out.get() # this may block
      parts_copies.append(his_parts) # increase the ``counter'' again

      print("Job finished at on problem",idx,flush=True)
      print("Local:",loss,posRate,negRate)
      validation_statistics[idx] = (loss,posRate,negRate)

    print("(Multi)-epoch",epoch,"validation finished at",time.time() - start_time)
    (loss,posRate,negRate) = np.mean(validation_statistics,axis=0)
    loss *= len(validation_statistics)# we want the loss summed up; so that mergeing preserves comparable values
    print("Validation stats:",loss,posRate,negRate,flush=True)

    valid_losses.append(loss)
    valid_posrates.append(posRate)
    valid_negrates.append(negRate)

    # plotting
    IC.plot_one("{}/plot.png".format(sys.argv[2]),times,train_losses,train_posrates,train_negrates,valid_losses,valid_posrates,valid_negrates)

  # a final "cleanup"
  for p in my_processes:
    p.kill()
