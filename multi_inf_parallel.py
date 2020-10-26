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

NUMPROCESSES = 25

MAX_ACTIVE_TASKS = NUMPROCESSES

DATA_THROUGH_QUEUE = False

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
  
  (loss_sum,posOK_sum,negOK_sum) = model()
  
  # print("Model evaluated")
  
  if training:
    loss_sum.backward()
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

  return (loss_sum[0].item(),posOK_sum,negOK_sum,tot_pos,tot_neg,myparts)

global common_data
common_data = None

def worker(q_in, q_out):
  global common_data
  while True:
    start_time = time.time()
    if DATA_THROUGH_QUEUE:
      (idx,data,myparts,training) = q_in.get()
    else:
      (idx,myparts,training) = q_in.get()
      metainfo,data = common_data[training][idx]
    
    (loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg,myparts) = eval_and_or_learn_on_one(myparts,data,training)
    q_out.put((idx,loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg,myparts,start_time,time.time()))

def get_size_from_idx(idx,actual_data_list):
  metainfo,data = actual_data_list[idx]
  return len(data[0])+len(data[1])

def big_go_last(feed_idx_sequence,actual_data_list):
  WHAT_IS_BIG = 10000

  big = [idx for idx in feed_idx_sequence if get_size_from_idx(idx,actual_data_list) > WHAT_IS_BIG]
  small = [idx for idx in feed_idx_sequence if get_size_from_idx(idx,actual_data_list) <= WHAT_IS_BIG]

  print("big_go_last",len(small),len(big))

  # big.sort() # start with the really big ones so that we are finished with them before the next iteration would be about to start
  return small #+big

def loop_it_out(start_time,t,parts_copies,feed_sequence,actual_data_list,training):
  stats = np.zeros(3) # loss_sum, posOK_sum, negOK_sum
  weights = np.zeros(2) # pos_weight, neg_weight

  # training on each problem in these EPOCHS_BEFORE_VALIDATION-many epochs
  while feed_sequence or len(parts_copies) < MAX_ACTIVE_TASKS:
    # we use parts_copies as a counter of idle children in the pool
    while parts_copies and feed_sequence:
      parts_copy = parts_copies.pop()
      idx = feed_sequence.pop()
      
      ((probname,probweight),data) = actual_data_list[idx]
      copy_parts_and_zero_grad_in_copy(master_parts,parts_copy)
      t += 1
      print(time.time() - start_time,"time_idx",t,"starting {} job on problem".format("training" if training else "validation"),idx,"of size",len(data[0])+len(data[1]),"and weight",probweight)
      
      if DATA_THROUGH_QUEUE:
        message = (idx,data,parts_copy,True) # True stands for "training is on"
      else:
        message = (idx,parts_copy,True) # True stands for "training is on"
      q_in.put(message)
      print(time.time() - start_time,"put finished",flush=True)
      print()

    print(time.time() - start_time,"about to call get")
    (idx,loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg,his_parts,time_start,time_end) = q_out.get() # this may block
    print(time.time() - start_time,"get finished")
    
    parts_copies.append(his_parts) # increase the ``counter'' again

    if training:
      copy_grads_back_from_param(master_parts,his_parts)
      print(time.time() - start_time,"copy_grads_back_from_param finished")
      optimizer.step()
      print(time.time() - start_time,"optimizer.step() finished")
    
    ((probname,probweight),data) = train_data_list[idx]
    pos_weight,neg_weight = data[-2],data[-1]

    print(time.time() - start_time,"job finished at on problem",idx,"started",time_start-start_time,"finished",time_end-start_time,"took",time_end-time_start,flush=True)
    print("Of weight",tot_pos,tot_neg,tot_pos+tot_neg)
    print("Debug",loss_sum,posOK_sum,negOK_sum)
    print("Local:",loss_sum/(tot_pos+tot_neg),posOK_sum/tot_pos if tot_pos > 0.0 else 1.0,negOK_sum/tot_neg if tot_neg > 0.0 else 1.0)
    print()

    stats += (loss_sum,posOK_sum,negOK_sum)
    weights += (pos_weight,neg_weight)

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
  # log = open("{}/run{}".format(sys.argv[2],IC.name_learning_regime_suffix()), 'w')
  # sys.stdout = log
  # sys.stderr = log
  
  start_time = time.time()
  
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

  print()
  print(time.time() - start_time,"Initialization finished")

  common_data = [valid_data_list,train_data_list]

  epoch = 0

  # in addition to the "oficial model" as named above, we checkpoint it as epoch0 here.
  model_name = "{}/model-epoch{}.pt".format(sys.argv[2],epoch)
  torch.save(master_parts,model_name)

  parts_copies = [] # have as many copies as MAX_ACTIVE_TASKS; they are somehow shared among the processes via Queue, so only one process should touch one at a time
  for i in range(MAX_ACTIVE_TASKS):
    parts_copies.append(torch.load(model_name)) # currently, don't know how to reliably deep-copy in memory (with pickling, all seems fine)

  q_in = torch.multiprocessing.Queue()
  q_out = torch.multiprocessing.Queue()
  my_processes = []
  for i in range(NUMPROCESSES):
    p = torch.multiprocessing.Process(target=worker, args=(q_in,q_out))
    p.start()
    my_processes.append(p)

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

  TRAIN_SAMPLES_PER_EPOCH = 1000
  VALID_SAMPLES_PER_EPOCH = 200

  t = 0

  while True:
    epoch += 1
   
    if epoch > 150:
      break
    
    times.append(epoch)

    train_feed_sequence = random.sample(range(len(train_data_list)),TRAIN_SAMPLES_PER_EPOCH) if len(train_data_list) > TRAIN_SAMPLES_PER_EPOCH else range(len(train_data_list)
    train_feed_sequence = big_go_last(train_feed_sequence,train_data_list) # largest go last, because loop_it_out pops from the end

    (t,stats,weights) = loop_it_out(start_time,t,parts_copies,train_feed_sequence,train_data_list,True) # True for training

    print()
    print("Epoch",epoch,"training finished at",time.time() - start_time)
    model_name = "{}/model-epoch{}.pt".format(sys.argv[2],epoch)
    print("Saving model to:",model_name)
    torch.save(master_parts,model_name)
    print()

    print("stats-weights",stats,weights)
    loss = stats[0]/(weights[0]+weights[1])
    posRate = stats[1]/weights[0]
    negRate = stats[2]/weights[1]
    print("Training stats:",loss,posRate,negRate,flush=True)

    train_losses.append(loss)
    train_posrates.append(posRate)
    train_negrates.append(negRate)

    print("Validating...")

    valid_feed_sequence = random.sample(range(len(valid_data_list)),VALID_SAMPLES_PER_EPOCH) if len(valid_data_list) > VALID_SAMPLES_PER_EPOCH else range(len(valid_data_list))
    valid_feed_sequence.sort(key=lambda idx : get_size_from_idx(idx,valid_data_list)) # largest go last, because loop_it_out pops from the end

    (t,stats,weights) = loop_it_out(start_time,t,parts_copies,valid_feed_sequence,valid_data_list,False) # False for evaluating

    print()
    print("Epoch",epoch,"validation finished at",time.time() - start_time)
  
    print("stats-weights",stats,weights)
    loss = stats[0]/(weights[0]+weights[1])
    posRate = stats[1]/weights[0]
    negRate = stats[2]/weights[1]
    print("Validation stats:",loss,posRate,negRate,flush=True)

    valid_losses.append(loss)
    valid_posrates.append(posRate)
    valid_negrates.append(negRate)

    # plotting
    IC.plot_one("{}/plot.png".format(sys.argv[2]),times,train_losses,train_posrates,train_negrates,valid_losses,valid_posrates,valid_negrates)

  # a final "cleanup"
  for p in my_processes:
    p.kill()
