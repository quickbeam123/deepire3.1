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
    (loss,posRate,negRate) = model()
  
    loss.backward()
    # put grad into actual tensor to be returned below (gradients don't go through the Queue)
    for param in myparts.parameters():
      grad = param.grad
      param.requires_grad = False # to allow the in-place operation just below
      if grad is not None:
        param.copy_(grad)
      else:
        param.zero_()
  
    result = (loss[0].detach().item(),posRate,negRate)
    
    torch.save(myparts,parts_file)
  
  else:
    with torch.no_grad():
      model = IC.LearningModel(*myparts,init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg)
      model.eval()
      (loss,posRate,negRate) = model()

      result = (loss[0].detach().item(),posRate,negRate)

  return result

def worker(q_in, q_out):

  # log = open("worker{}.log".format(os.getpid()), 'w')
  log = sys.stdout

  # from guppy import hpy
  # h = hpy()

  '''
  import tracemalloc
  tracemalloc.start(10)  # save upto 5 stack frames
  
  time1 = tracemalloc.take_snapshot()
  time2 = None
  '''

  while True:
    (probname,parts_file,training) = q_in.get()
    
    start_time = time.time()
    (loss,posRate,negRate) = eval_and_or_learn_on_one(probname,parts_file,training,log)
    q_out.put((probname,loss,posRate,negRate,parts_file,start_time,time.time()))

    '''
    cnt = 0
    sizes = 0
    for tracked_object in gc.get_objects():
      cnt += 1
      sizes += sys.getsizeof(tracked_object)
    print("begofeGC",cnt,"objects of total size",sizes,file=log,flush=True)

    stat = gc.collect()
    
    print("collected",stat,file=f,flush=True)
    
    cnt = 0
    sizes = 0
    for tracked_object in gc.get_objects():
      cnt += 1
      sizes += sys.getsizeof(tracked_object)
    print("afterGC",cnt,"objects of total size",sizes,file=log,flush=True)
    '''

    '''
    if time2 is not None:
      print("Diff to prev",file=log,flush=True)
      new_time = tracemalloc.take_snapshot()
      stats = new_time.compare_to(time2, 'lineno')
      for stat in stats[:10]:
        print(stat,file=f)
      stats = new_time.compare_to(time2, 'traceback')
      top = stats[0]
      print('\n'.join(top.traceback.format()),file=f,flush=True)
      time2 = new_time
    else:
      time2 = tracemalloc.take_snapshot()

    print("\nDiff to base",file=log,flush=True)
    stats = time2.compare_to(time1, 'lineno')
    for stat in stats[:10]:
      print(stat,file=f)
    stats = time2.compare_to(time1, 'traceback')
    top = stats[0]
    print('\n'.join(top.traceback.format()),file=log,flush=True)
    '''
    # print(h.heap(),file=log,flush=True)

def big_go_last(feed_sequence):
  WHAT_IS_BIG = 6000

  big = [(size,piece_name) for (size,piece_name) in feed_sequence if size > WHAT_IS_BIG]
  small = [(size,piece_name) for (size,piece_name) in feed_sequence if size <= WHAT_IS_BIG]

  print("big_go_last",len(small),len(big))

  big.sort() # start with the really big ones so that we are finished with them before the next iteration would be about to start

  return small+big

def loop_it_out(start_time,t,feed_sequence,training):
  MAX_ACTIVE_TASKS = NUMPROCESSES
  
  num_active_tasks = 0
  
  statistics = []
  
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
      print(time.time() - start_time,"put finished")
      print()

    print(time.time() - start_time,"about to call get")
    (probname,loss,posRate,negRate,parts_file,time_start,time_end) = q_out.get() # this may block
    print(time.time() - start_time,"get finished")
    
    num_active_tasks -= 1
    if training:
      his_parts = torch.load(parts_file)
      os.remove(parts_file)
      copy_grads_back_from_param(master_parts,his_parts)
      print(time.time() - start_time,"copy_grads_back_from_param finished")
      optimizer.step()
      print(time.time() - start_time,"optimizer.step() finished")

    print(time.time() - start_time,"job finished at on problem",probname,"started",time_start-start_time,"finished",time_end-start_time,"took",time_end-time_start,flush=True)
    print("Local:",loss,posRate,negRate)
    print()
    statistics.append((loss,posRate,negRate))
  
    cnt = 0
    sizes = 0
    for tracked_object in gc.get_objects():
      cnt += 1
      sizes += sys.getsizeof(tracked_object)
    print("gc-info",cnt,"objects of total size",sizes,flush=True)
  
  if not training:
    os.remove(parts_file)
    
  return (t,statistics)

  '''
  print("gc.get_stats()",gc.get_stats())
  cnt = 0
  sizes = 0
  for tracked_object in gc.get_objects():
    cnt += 1
    sizes += sys.getsizeof(tracked_object)
  print("begofeGC",cnt,"objects of total size",sizes,flush=True)

  stat = gc.collect()
  
  print("collected",stat,flush=True)
  
  cnt = 0
  sizes = 0
  for tracked_object in gc.get_objects():
    cnt += 1
    sizes += sys.getsizeof(tracked_object)
  print("afterGC",cnt,"objects of total size",sizes,flush=True)
  '''

  '''
  if time2 is not None:
    print("Diff to prev",flush=True)
    new_time = tracemalloc.take_snapshot()
    stats = new_time.compare_to(time2, 'lineno')
    for stat in stats[:10]:
      print(stat)
    stats = new_time.compare_to(time2, 'traceback')
    top = stats[0]
    print('\n'.join(top.traceback.format()))
    time2 = new_time
  else:
    time2 = tracemalloc.take_snapshot()

  print("\nDiff to base")
  stats = time2.compare_to(time1, 'lineno')
  for stat in stats[:10]:
    print(stat)
  stats = time2.compare_to(time1, 'traceback')
  top = stats[0]
  print('\n'.join(top.traceback.format()))
  '''

  # print(h.heap())

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
    train_data_idx += valid_data_idx
    valid_data_idx = []
    print("Merged valid with train; final:",len(train_data_idx))

  print()
  print(time.time() - start_time,"Initialization finished")

  epoch = 0
  # in addition to the "oficial model" as named above, we checkpoint it as epoch0 here.
  model_name = "{}/model-epoch{}.pt".format(sys.argv[2],epoch)
  torch.save(master_parts,model_name)

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

  # from guppy import hpy
  # h = hpy()

  '''
  import tracemalloc
  tracemalloc.start(10)  # save upto 5 stack frames
  time1 = tracemalloc.take_snapshot()
  time2 = None
  '''

  gc.set_debug(gc.DEBUG_LEAK | gc.DEBUG_STATS)

  TRAIN_SAMPLES_PER_EPOCH = 800
  VALID_SAMPLES_PER_EPOCH = 200

  t = 0

  while True:
    epoch += 1
   
    if epoch > 150:
      break
    
    times.append(epoch)

    feed_sequence = random.sample(train_data_idx,TRAIN_SAMPLES_PER_EPOCH)
    feed_sequence = big_go_last(feed_sequence) # largest go last, because loop_it_out pops from the end

    (t,statistics) = loop_it_out(start_time,t,feed_sequence,True) # True for training

    print("Epoch",epoch,"traing finished at",time.time() - start_time)
    model_name = "{}/model-epoch{}.pt".format(sys.argv[2],epoch)
    print("Saving model to:",model_name)
    torch.save(master_parts,model_name)
    print()
    
    (loss,posRate,negRate) = np.mean(np.array(statistics),axis=0)
    loss *= len(statistics) # we want the loss summed up; so that mergeing preserves comparable values
    print("Training stats:",loss,posRate,negRate,flush=True)
    print()
    
    train_losses.append(loss)
    train_posrates.append(posRate)
    train_negrates.append(negRate)

    print("Validating...")

    feed_sequence = random.sample(valid_data_idx,VALID_SAMPLES_PER_EPOCH)
    feed_sequence.sort() # largest go last, because loop_it_out pops from the end

    (t,statistics) = loop_it_out(start_time,t,feed_sequence,False) # False for evaluating

    print("Epoch",epoch,"validation finished at",time.time() - start_time)
    (loss,posRate,negRate) = np.mean(np.array(statistics),axis=0)
    loss *= len(statistics) # we want the loss summed up; so that mergeing preserves comparable values
    print("Validation stats:",loss,posRate,negRate,flush=True)

    valid_losses.append(loss)
    valid_posrates.append(posRate)
    valid_negrates.append(negRate)

    # plotting
    IC.plot_one("{}/plot.png".format(sys.argv[2]),times,train_losses,train_posrates,train_negrates,valid_losses,valid_posrates,valid_negrates)

  # a final "cleanup"
  for p in my_processes:
    p.kill()
