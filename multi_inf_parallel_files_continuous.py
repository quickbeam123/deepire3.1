#!/usr/bin/env python3

import os

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

import sys,random,itertools,os,gc

import numpy as np

# To release claimed memory back to os; Call:   libc.malloc_trim(ctypes.c_int(0))
import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

def copy_grads_back_from_param(parts,parts_copies):
  for param, param_copy in zip(parts.parameters(),parts_copies.parameters()):
    # print("Copy",param_copy)
    # print("Copy.grad",param_copy.grad)
    param.grad = param_copy

def eval_and_or_learn_on_one(probname,parts_file,training,log):

  # print("eval_and_or_learn_on_one",probname,parts_file,training)
  # print("{}/pieces/{}".format(sys.argv[1],probname))

  data = torch.load("{}/pieces/{}".format(sys.argv[1],probname))
  (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) = data

  myparts = torch.load(parts_file)
  
  # not sure if there is any after load -- TODO: check if necessary
  for param in myparts.parameters():
    # taken from Optmizier zero_grad, roughly
    if param.grad is not None:
      print("Loaded param with with a grad",log)
      param.grad.detach_()
      param.grad.zero_()

  # print("Datum of size",len(init)+len(deriv))

  if training:
    model = IC.LearningModel(*myparts,init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg)
    model.train()
    (loss_sum,posOK_sum,negOK_sum) = model()
  
    loss_sum.backward()
    # put grad into actual tensor to be returned below (gradients don't go through the Queue)
    for param in myparts.parameters():
      grad = param.grad
      param.requires_grad = False # to allow the in-place operation just below
      if grad is not None:
        param.copy_(grad)
      else:
        param.zero_()
  
    torch.save(myparts,parts_file)
  
  else:
    with torch.no_grad():
      model = IC.LearningModel(*myparts,init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg)
      model.eval()
      (loss_sum,posOK_sum,negOK_sum) = model()

  del model # I am getting desperate!

  return (loss_sum[0].detach().item(),posOK_sum,negOK_sum,tot_pos,tot_neg)

def worker(q_in, q_out):
  log = sys.stdout

  while True:
    (probname,size,parts_file,training) = q_in.get()
    
    start_time = time.time()
    (loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg) = eval_and_or_learn_on_one(probname,parts_file,training,log)
    q_out.put((probname,size,loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg,parts_file,start_time,time.time()))

    libc.malloc_trim(ctypes.c_int(0))

def save_checkpoint(epoch, model, optimizer):
  print("checkpoint",epoch)

  check_name = "{}/check-epoch{}.pt".format(sys.argv[2],epoch)
  check = (epoch,model,optimizer)
  torch.save(check,check_name)

def load_checkpoint(filename):
  return torch.load(filename)

def weighted_std_deviation(weighted_mean,scaled_values,weights,weight_sum):
  # print(weighted_mean)
  # print(scaled_values)
  # print(weights)
  # print(weight_sum,flush=True)

  values = np.divide(scaled_values, weights, out=np.ones_like(scaled_values), where=weights!=0.0) # whenever the respective weight is 0.0, the result should be understood as 1.0
  squares = (values - weighted_mean)**2
  std_dev = np.sqrt(np.sum(weights*squares,axis=0) / weight_sum)
  return std_dev

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
  log = open("{}/run{}".format(sys.argv[2],IC.name_learning_regime_suffix()), 'w')
  sys.stdout = log
  sys.stderr = log
  
  start_time = time.time()
  
  train_data_idx = torch.load("{}/training_index.pt".format(sys.argv[1]))
  print("Loaded train data:",len(train_data_idx))
  valid_data_idx = torch.load("{}/validation_index.pt".format(sys.argv[1]))
  print("Loaded valid data:",len(valid_data_idx))
  
  if HP.TRR == HP.TestRiskRegimen_OVERFIT:
    # merge validation data back to training set (and ditch early stopping regularization)
    train_data_idx += valid_data_idx
    valid_data_idx = []
    print("Merged valid with train; final:",len(train_data_idx))

  print()
  print(time.time() - start_time,"Initialization finished",flush=True)

  epoch = 0

  MAX_EPOCH = HP.MAX_EPOCH
  
  if len(sys.argv) >= 4:
    (epoch,master_parts,optimizer) = load_checkpoint(sys.argv[3])
    print("Loaded checkpoint",sys.argv[3],flush=True)
  
    if len(sys.argv) >= 5:
      MAX_EPOCH = int(sys.argv[4])
  
    # update the learning rate according to hyperparams
    for param_group in optimizer.param_groups:
        param_group['lr'] = HP.LEARN_RATE
    print("Set optimizer's (nominal) learning rate to",HP.LEARN_RATE)
  else:
    thax_sign,sine_sign,deriv_arits,thax_to_str = torch.load("{}/data_sign.pt".format(sys.argv[1]))
    master_parts = IC.get_initial_model(thax_sign,sine_sign,deriv_arits)
    model_name = "{}/initial{}".format(sys.argv[2],IC.name_initial_model_suffix())
    torch.save(master_parts,model_name)
    print("Created model parts and saved to",model_name,flush=True)
    
    if HP.OPTIMIZER == HP.Optimizer_SGD: # could also play with momentum and its dampening here!
      optimizer = torch.optim.SGD(master_parts.parameters(), lr=HP.LEARN_RATE)
    elif HP.OPTIMIZER == HP.Optimizer_ADAM:
      optimizer = torch.optim.Adam(master_parts.parameters(), lr=HP.LEARN_RATE, weight_decay=HP.WEIGHT_DECAY)
    elif  HP.OPTIMIZER == HP.Optimizer_ADAMW:
      optimizer = torch.optim.AdamW(master_parts.parameters(), lr=HP.LEARN_RATE, weight_decay=HP.WEIGHT_DECAY)

  q_in = torch.multiprocessing.Queue()
  q_out = torch.multiprocessing.Queue()
  my_processes = []
  for i in range(HP.NUMPROCESSES):
    p = torch.multiprocessing.Process(target=worker, args=(q_in,q_out))
    p.start()
    my_processes.append(p)

  times = []
  losses = []
  losses_devs = []
  posrates = []
  posrates_devs = []
  negrates = []
  negrates_devs = []

  id = 0 # id synchronizes writes to the worker pipe

  samples_per_epoch = len(train_data_idx)

  t = epoch*samples_per_epoch # time synchronizes writes to master_parts and the stasts

  stats = np.zeros((samples_per_epoch,3)) # loss_sum, posOK_sum, negOK_sum
  weights = np.zeros((samples_per_epoch,2)) # pos_weight, neg_weight

  MAX_ACTIVE_TASKS = HP.NUMPROCESSES
  num_active_tasks = 0
  
  # NOTE: on air05, 3050189 was swapping (observed with thax500, embeddings 256)
  
  # NOTE: on air05, 4000000 was still not enough for a good flow, but we got unstuck aumotmatically after a swapping slowdown
  # (observed with thax2000, embeddings 256)
   
  # with 3500000 there was still a slowdown (thax2000, emb 256) probably cause by a swapping period?

  MAX_CAPACITY = 5000000 # a total size of 4653978 caused a crash on air05 with 300G RAM (maybe air04 would still be able to cope with 5M?)
  # note that that was a run on thax1000, i.e. 1000 embeddings of axioms (how much do these actually take up in comparison to the proper matrices?)
  assert HP.NUMPROCESSES * HP.COMPRESSION_THRESHOLD * 5 // 4 < MAX_CAPACITY
  cur_allocated = 0

  while True:
    while num_active_tasks < MAX_ACTIVE_TASKS:
      num_active_tasks += 1
      
      while True:
        (size,probname) = random.choice(train_data_idx)
        print("Picking",probname,"of size",size,end="...")
        
        if size >= HP.WHAT_IS_HUGE:
          print("Is huge, skipping.")
        elif cur_allocated + size > MAX_CAPACITY - (MAX_ACTIVE_TASKS-num_active_tasks) * (HP.COMPRESSION_THRESHOLD * 5 // 4):
          print(f"too big! (cur_allocated is {cur_allocated} and still {(MAX_ACTIVE_TASKS-num_active_tasks)} tasks need to allocate)")
        else:
          print()
          break
      cur_allocated += size
    
      print(time.time() - start_time,"starting job on problem",probname,"of size",size,flush=True)
      
      id += 1
      parts_file = "{}/parts_{}_{}.pt".format(HP.SCRATCH,os.getpid(),id)
      torch.save(master_parts,parts_file)
      
      q_in.put((probname,size,parts_file,True)) # training is always true in continuous! (TODO: factor out worker to inf_common!)
      print(time.time() - start_time,"put finished",flush=True)
      print()

    print(time.time() - start_time,"about to call get")

    (probname,size,loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg,parts_file,time_start,time_end) = q_out.get() # this may block

    cur_allocated -= size

    print(time.time() - start_time,"job finished at on problem",probname,"started",time_start-start_time,"finished",time_end-start_time,"took",time_end-time_start,flush=True)
    print("Of weight",tot_pos,tot_neg,tot_pos+tot_neg)
    print("Loss,pos,neg:",loss_sum/(tot_pos+tot_neg),posOK_sum/tot_pos if tot_pos > 0.0 else 1.0,negOK_sum/tot_neg if tot_neg > 0.0 else 1.0)
    print()

    stats[t % samples_per_epoch] = (loss_sum,posOK_sum,negOK_sum)
    weights[t % samples_per_epoch] = (tot_pos,tot_neg)

    t += 1
    print(time.time() - start_time,"get finished for time_idx",t)
  
    if HP.NON_CONSTANT_10_50_250_LR:
      # LATER NORMALIZE THIS:
      # it worked well with 256 embedding size
      # it worked well with 40 processes active at a time!
      if t <= 50*samples_per_epoch: # initial warmup: take "50 000" optimizer steps (= 50 epochs) to reach 5*HP.LEARN_RATE (in 10 epochs, HP.LEARN_RATE has been reached and then it's gradually overshot)
        lr = HP.LEARN_RATE*t/(10*samples_per_epoch)
        print("Increasing effective LR to",lr,flush=True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
      else: # hyperbolic cooldown (reach HP.LEARN_RATE at "250 000" = 250 epochs)
        lr = 250*samples_per_epoch/t*HP.LEARN_RATE
        print("Dropping effective LR to",lr,flush=True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    num_active_tasks -= 1
    his_parts = torch.load(parts_file)
    os.remove(parts_file)
    copy_grads_back_from_param(master_parts,his_parts)
    print(time.time() - start_time,"copy_grads_back_from_param finished")
    if HP.CLIP_GRAD_NORM:
      torch.nn.utils.clip_grad_norm_(master_parts.parameters(), HP.CLIP_GRAD_NORM)
    if HP.CLIP_GRAD_VAL:
      torch.nn.utils.clip_grad_value_(master_parts.parameters(), HP.CLIP_GRAD_VAL)
    
    optimizer.step()
    print(time.time() - start_time,"optimizer.step() finished")

    if t % samples_per_epoch == 0:
      epoch += 1
    
      print("Epoch",epoch,"finished at",time.time() - start_time)
      save_checkpoint(epoch,master_parts,optimizer)
      print()

      # print("stats",stats)
      # print("weights",weights)

      # sum-up stats over the "samples_per_epoch" entries (retain the var name):
      loss_sum,posOK_sum,negOK_sum = np.sum(stats,axis=0)
      tot_pos,tot_neg = np.sum(weights,axis=0)
    
      print("loss_sum,posOK_sum,negOK_sum",loss_sum,posOK_sum,negOK_sum)
      print("tot_pos,tot_neg",tot_pos,tot_neg)

      # CAREFULE: could divide by zero!
      sum_stats = np.sum(stats,axis=0)
      loss = sum_stats[0]/(tot_pos+tot_neg)
      posrate = sum_stats[1]/tot_pos
      negrate = sum_stats[2]/tot_neg

      loss_dev = weighted_std_deviation(loss,stats[:,0],np.sum(weights,axis=1),tot_pos+tot_neg)
      posrate_dev = weighted_std_deviation(posrate,stats[:,1],weights[:,0],tot_pos)
      negrate_dev = weighted_std_deviation(negrate,stats[:,2],weights[:,1],tot_neg)
      
      print("Training stats:")
      print("Loss:",loss,"+/-",loss_dev,flush=True)
      print("Posrate:",posrate,"+/-",posrate_dev,flush=True)
      print("Negrate:",negrate,"+/-",negrate_dev,flush=True)
      print()
    
      times.append(epoch)
      losses.append(loss)
      losses_devs.append(loss_dev)
      posrates.append(posrate)
      posrates_devs.append(posrate_dev)
      negrates.append(negrate)
      negrates_devs.append(negrate_dev)
      
      IC.plot_with_devs("{}/plot.png".format(sys.argv[2]),times,losses,losses_devs,posrates,posrates_devs,negrates,negrates_devs)
      
      if epoch >= MAX_EPOCH:
        break

  # a final "cleanup"
  for p in my_processes:
    p.kill()
