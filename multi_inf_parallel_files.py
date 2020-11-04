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

NUMPROCESSES = 25

SCRATCH = "/scratch/sudamar2/"

def copy_grads_back_from_param(parts,parts_copies):
  for param, param_copy in zip(parts.parameters(),parts_copies.parameters()):
    # print("Copy",param_copy)
    # print("Copy.grad",param_copy.grad)
    param.grad = param_copy

def eval_and_or_learn_on_one(probname,parts_file,training,log):

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
    (probname,parts_file,training) = q_in.get()
    
    start_time = time.time()
    (loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg) = eval_and_or_learn_on_one(probname,parts_file,training,log)
    q_out.put((probname,loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg,parts_file,start_time,time.time()))

    libc.malloc_trim(ctypes.c_int(0))

def big_go_last(feed_sequence):
  huge = [(size,piece_name) for (size,piece_name) in feed_sequence if size > HP.WHAT_IS_HUGE]
  big = [(size,piece_name) for (size,piece_name) in feed_sequence if size > HP.WHAT_IS_BIG and size <= HP.WHAT_IS_HUGE]
  small = [(size,piece_name) for (size,piece_name) in feed_sequence if size <= HP.WHAT_IS_BIG]

  print("big_go_last",len(small),len(big),len(huge))

  big.sort() # start with the really big ones so that we are finished with them before the next iteration would be about to start
  return small+big

def loop_it_out(start_time,t,feed_sequence,optimizer,scheduler,training):
  MAX_ACTIVE_TASKS = NUMPROCESSES
  
  num_active_tasks = 0
  
  stats = np.zeros(3) # loss_sum, posOK_sum, negOK_sum
  weights = np.zeros(2) # pos_weight, neg_weight
  
  if not training:
    parts_file = "{}/master_{}.pt".format(SCRATCH,os.getpid())
    torch.save(master_parts,parts_file)
  
  while feed_sequence or num_active_tasks > 0:
    # we use parts_copies as a counter of idle children in the pool
    while num_active_tasks < MAX_ACTIVE_TASKS and feed_sequence:
      t += 1
      num_active_tasks += 1
      (size,probname) = feed_sequence.pop()
      
      print(time.time() - start_time,"time_idx",t,"starting {} job on problem".format("training" if training else "validation"),probname,"of size",size)
      
      if training:
        parts_file = "{}/parts_{}_{}.pt".format(SCRATCH,os.getpid(),t)
        torch.save(master_parts,parts_file)
        print(time.time() - start_time,"parts saved")
          
      q_in.put((probname,parts_file,training))
      print(time.time() - start_time,"put finished",flush=True)
      print()

    print(time.time() - start_time,"about to call get")
    (probname,loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg,parts_file,time_start,time_end) = q_out.get() # this may block
    print(time.time() - start_time,"get finished")
    
    num_active_tasks -= 1
    if training:
      his_parts = torch.load(parts_file)
      os.remove(parts_file)
      copy_grads_back_from_param(master_parts,his_parts)
      print(time.time() - start_time,"copy_grads_back_from_param finished")
      optimizer.step()
      if scheduler:
        scheduler.step()
      print(time.time() - start_time,"optimizer.step() finished")

    print(time.time() - start_time,"job finished at on problem",probname,"started",time_start-start_time,"finished",time_end-start_time,"took",time_end-time_start,flush=True)
    print("Of weight",tot_pos,tot_neg,tot_pos+tot_neg)
    print("Debug",loss_sum,posOK_sum,negOK_sum)
    print("Local:",loss_sum/(tot_pos+tot_neg),posOK_sum/tot_pos if tot_pos > 0.0 else 1.0,negOK_sum/tot_neg if tot_neg > 0.0 else 1.0)
    print()

    stats += (loss_sum,posOK_sum,negOK_sum)
    weights += (tot_pos,tot_neg)
  
    cnt = 0
    sizes = 0
    for tracked_object in gc.get_objects():
      cnt += 1
      sizes += sys.getsizeof(tracked_object)
    print("gc-info",cnt,"objects of total size",sizes,flush=True)
  
  if not training:
    os.remove(parts_file)
    
  return (t,stats,weights)

def save_checkpoint(epoch, model, optimizer):
  print("checkpoint",epoch)

  check_name = "{}/check-epoch{}.pt".format(sys.argv[2],epoch)
  check = (epoch,model,optimizer)
  torch.save(check,check_name)

def load_checkpoint(filename):
  return torch.load(filename)

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
  print(time.time() - start_time,"Initialization finished")

  epoch = 0
  
  if len(sys.argv) >= 4:
    (epoch,master_parts,optimizer) = load_checkpoint(sys.argv[3])
    print("Loaded checkpoint",sys.argv[3])
  else:
    thax_sign,sine_sign,deriv_arits,thax_to_str = torch.load("{}/data_sign.pt".format(sys.argv[1]))
    master_parts = IC.get_initial_model(thax_sign,sine_sign,deriv_arits)
    model_name = "{}/initial{}".format(sys.argv[2],IC.name_initial_model_suffix())
    torch.save(master_parts,model_name)
    print("Created model parts and saved to",model_name)
    
    if HP.OPTIMIZER == HP.Optimizer_SGD: # could also play with momentum and its dampening here!
      optimizer = torch.optim.SGD(master_parts.parameters(), lr=HP.LEARN_RATE)
    elif HP.OPTIMIZER == HP.Optimizer_ADAM:
      optimizer = torch.optim.Adam(master_parts.parameters(), lr=HP.LEARN_RATE)

    save_checkpoint(epoch,master_parts,optimizer)

  if False: # TODO: make this a HP entry?
    MAX_LR = 0.01
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=MAX_LR,steps_per_epoch=len(train_data_idx),epochs=50)
  else:
    scheduler = None

  q_in = torch.multiprocessing.Queue()
  q_out = torch.multiprocessing.Queue()
  my_processes = []
  for i in range(NUMPROCESSES):
    p = torch.multiprocessing.Process(target=worker, args=(q_in,q_out))
    p.start()
    my_processes.append(p)

  times = []
  # rates = []
  train_losses = []
  train_posrates = []
  train_negrates = []
  valid_losses = []
  valid_posrates = []
  valid_negrates = []

  # from guppy import hpy
  # h = hpy()

  '''
  import tracemalloc
  tracemalloc.start(10)  # save upto 5 stack frames
  time1 = tracemalloc.take_snapshot()
  time2 = None
  '''

  # gc.set_debug(gc.DEBUG_LEAK | gc.DEBUG_STATS)

  TRAIN_SAMPLES_PER_EPOCH = 1000
  VALID_SAMPLES_PER_EPOCH = 200

  # temporarily train / validate just on a fixed subset
  '''
  train_feed_sequence = random.sample(train_data_idx,TRAIN_SAMPLES_PER_EPOCH)
  train_feed_sequence = big_go_last(train_feed_sequence) # largest go last, because loop_it_out pops from the end
  valid_feed_sequence = random.sample(valid_data_idx,VALID_SAMPLES_PER_EPOCH)
  valid_feed_sequence.sort() # largest go last, because loop_it_out pops from the end
  careful, use with copy in the call to loop_it_out
  '''

  t = 0

  while True:
    epoch += 1
   
    if epoch > 50:
      break
  
    times.append(epoch)
    # rates.append(scheduler.get_last_lr()[0]/MAX_LR)

    '''
    lr = 0.0001*epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    times.append(lr)
    '''

    train_feed_sequence = random.sample(train_data_idx,TRAIN_SAMPLES_PER_EPOCH) if len(train_data_idx) > TRAIN_SAMPLES_PER_EPOCH else train_data_idx.copy()
    train_feed_sequence = big_go_last(train_feed_sequence) # largest go last, because loop_it_out pops from the end

    (t,stats,weights) = loop_it_out(start_time,t,train_feed_sequence,optimizer,scheduler,True) # True for training

    print("Epoch",epoch,"training finished at",time.time() - start_time)
    save_checkpoint(epoch,master_parts,optimizer)
    print()
    
    print("stats-weights",stats,weights)
    loss = stats[0]/(weights[0]+weights[1])
    posRate = stats[1]/weights[0]
    negRate = stats[2]/weights[1]
    print("Training stats:",loss,posRate,negRate,flush=True)
    print()
    
    train_losses.append(loss)
    train_posrates.append(posRate)
    train_negrates.append(negRate)

    print("Validating...")

    valid_feed_sequence = random.sample(valid_data_idx,VALID_SAMPLES_PER_EPOCH) if len(valid_data_idx) > VALID_SAMPLES_PER_EPOCH else valid_data_idx.copy()
    valid_feed_sequence.sort() # largest go last, because loop_it_out pops from the end

    (t,stats,weights) = loop_it_out(start_time,t,valid_feed_sequence,optimizer,scheduler,False) # False for evaluating

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
