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

import sys,random,itertools,os,gc,errno

import numpy as np

import matplotlib.pyplot as plt

# To release claimed memory back to os; Call:   libc.malloc_trim(ctypes.c_int(0))
import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

# models are global (anyway just read-only) to all workers (does not work on mac)

def eval_on_one(my_parts,piece_name):
  data = torch.load("{}/pieces/{}".format(sys.argv[1],piece_name))
  (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) = data
  
  with torch.no_grad():
    model = IC.LearningModel(*my_parts,init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg)
    model.eval()
    (loss_sum,posOK_sum,negOK_sum) = model()

  return (loss_sum[0].detach().item(),posOK_sum,negOK_sum,tot_pos,tot_neg)

def worker(q_in, q_out):
  log = sys.stdout

  my_parts = None
  my_model_path = None

  while True:
    (model_path,piece_name) = q_in.get()
    
    if my_model_path != model_path:
      (epoch,parts,optimizer) = torch.load(model_path)
      my_parts = parts
      my_model_path = model_path
  
    start_time = time.time()
    (loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg) = eval_on_one(my_parts,piece_name)
    
    q_out.put((piece_name,loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg,start_time,time.time()))

    libc.malloc_trim(ctypes.c_int(0))

def weighted_std_deviation(weighted_mean,scaled_values,weights,weight_sum):
  values = np.divide(scaled_values, weights, out=np.ones_like(scaled_values), where=weights!=0.0) # whenever the respective weight is 0.0, the result should be understood as 1.0
  squares = (values - weighted_mean)**2
  std_dev = np.sqrt(np.sum(weights*squares,axis=0) / weight_sum)
  return std_dev

def plot_summary_and_report_best(datapoints):
  times = []
  losses = []
  losses_devs = []
  posrates = []
  posrates_devs = []
  negrates = []
  negrates_devs = []

  best_loss = 1000.0
  best_idx = None

  for i,(model_num,loss,loss_dev,posrate,posrate_dev,negrate,negrate_dev) in enumerate(sorted(datapoints)):
    if loss < best_loss:
      best_loss = loss
      best_idx = i

    times.append(model_num)
    losses.append(loss)
    losses_devs.append(loss_dev)
    posrates.append(posrate)
    posrates_devs.append(posrate_dev)
    negrates.append(negrate)
    negrates_devs.append(negrate_dev)

  print("Best validation loss model:",times[best_idx],end=" ")
  print("loss",losses[best_idx],"posrate",posrates[best_idx],"negrate",negrates[best_idx])

  IC.plot_with_devs("{}/plot.png".format(sys.argv[3]),times,losses,losses_devs,posrates,posrates_devs,negrates,negrates_devs)

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Load a bunch of checkpoints and start properly evaluating them on train/valid/both
  #
  # To be called as in ./validator.py <folder1> <folder2> <folder3>
  #
  # Expects <folder1> to contain "validation_data.pt" and the corresponding ./points/ subfolder
  #
  # <folder2> to contain files "check-epoch<number>.pt"
  #
  # The log, the converging plots, are to be saved to <folder3>
  
  # Careful -- ANNOINGLY, THE EVALUATION DEPENDS on HP.POS_WEIGHT_EXTRA
  
  # This tool is meant to run in sync with training, so it will always look for the latest checkpoint it hasn't evaluated yet!
  
  # global redirect of prints to the just open "logfile"
  log = open("{}/validate{}".format(sys.argv[3],IC.name_learning_regime_suffix()), 'w')
  sys.stdout = log
  sys.stderr = log
  
  # DATA PREPARATION
  
  valid_data_idx = torch.load("{}/validation_index.pt".format(sys.argv[1]))
  print("Loaded valid data:",len(valid_data_idx))
  
  # so that the largest ones start evaluating first
  valid_data_idx.sort() # we pop (from the end), so no need for reverse=True!
  
  # don't validate on the huge ones. It eats crazy amounts of memory:
  while valid_data_idx[-1][0] > HP.WHAT_IS_HUGE:
    print("Dropping too huge:",valid_data_idx.pop())
    
  # valid_data_idx = valid_data_idx[:10] # just to debug
  
  print(flush=True)

  # LOAD DATAPOINTS if ALREADY HAVING SOME
  
  evaluated_models = set()
  datapoints = [] # a list of tuples (model_num,loss,loss_dev,posrate,posrate_dev,negrate,negrate_dev)

  dir = "{}/points/".format(sys.argv[3])
  try:
    os.mkdir(dir)
  except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

  for filename in os.listdir(dir):
    if filename.startswith("datapoint-") and filename.endswith(".pt"):
      print("Found saved datapoint",filename)

      datapoint = torch.load(dir+filename)
      model_num = datapoint[0]
      if model_num in evaluated_models:
        print("Already known",filename,"Skipping...")
      else:
        evaluated_models.add(model_num)
        datapoints.append(datapoint)

  if len(datapoints) > 0:
    plot_summary_and_report_best(datapoints)

  MAX_ACTIVE_TASKS = 30
  num_active_tasks = 0

  q_in = torch.multiprocessing.Queue()
  q_out = torch.multiprocessing.Queue()
  my_processes = []
  for i in range(MAX_ACTIVE_TASKS):
    p = torch.multiprocessing.Process(target=worker, args=(q_in,q_out))
    p.start()
    my_processes.append(p)

  # LOOPING
  while True:
    youngest_unevaluated = None
    youngests_age = 0
    # Look for a not yet evaluated check-point
    for filename in os.listdir(sys.argv[2]):
      if filename.startswith("check-epoch") and filename.endswith(".pt"):
        model_num = int(filename[11:-3])
        # print(filename,model_num)
        if model_num not in evaluated_models and model_num > youngests_age:
          # print("Update",youngest_unevaluated,youngests_age)
          youngest_unevaluated = filename
          youngests_age = model_num

    if youngest_unevaluated:
      print("Selected youngest_unevaluated",youngest_unevaluated)
      model_path = sys.argv[2]+"/"+youngest_unevaluated
      print(model_path)
    else:
      print("Could not find unevaluated checkpoint. Exiting.")
      break

    print()
    print("Starting timer",flush=True)
    start_time = time.time()

    feed_sequence = valid_data_idx.copy()
    stats = np.zeros((len(feed_sequence),3)) # loss_sum, posOK_sum, negOK_sum
    weights = np.zeros((len(feed_sequence),2)) # pos_weight, neg_weight
    t = 0

    while feed_sequence or num_active_tasks > 0:
      # we use parts_copies as a counter of idle children in the pool
      while num_active_tasks < MAX_ACTIVE_TASKS and feed_sequence:
        
        num_active_tasks += 1
        (size,piece_name) = feed_sequence.pop()
        
        print(time.time() - start_time,"starting job on problem",piece_name,"of size",size,flush=True)
        q_in.put((model_path,piece_name))
        
        print()

      (piece_name,loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg,time_start,time_end) = q_out.get() # this may block
      
      num_active_tasks -= 1

      print(time.time() - start_time,"job finished at on problem",piece_name,"started",time_start-start_time,"finished",time_end-start_time,"took",time_end-time_start,flush=True)
      print("Of weight",tot_pos,tot_neg,tot_pos+tot_neg)
      print("Loss,pos,neg:",loss_sum/(tot_pos+tot_neg),posOK_sum/tot_pos if tot_pos > 0.0 else 1.0,negOK_sum/tot_neg if tot_neg > 0.0 else 1.0)
      print()

      stats[t] = (loss_sum,posOK_sum,negOK_sum)
      weights[t] = (tot_pos,tot_neg)
      t += 1

    # finished inner loop
    print("Validation",youngest_unevaluated,"finished at",time.time() - start_time)
    print()

    # sum-up stats over the SAMPLES_PER_EPOCH entries (retain the var name):
    loss_sum,posOK_sum,negOK_sum = np.sum(stats,axis=0)
    tot_pos,tot_neg = np.sum(weights,axis=0)
    
    print("loss_sum,posOK_sum,negOK_sum",loss_sum,posOK_sum,negOK_sum)
    print("tot_pos,tot_neg",tot_pos,tot_neg)

    # CAREFUL: could divide by zero!
    sum_stats = np.sum(stats,axis=0)
    loss = sum_stats[0]/(tot_pos+tot_neg)
    posrate = sum_stats[1]/tot_pos
    negrate = sum_stats[2]/tot_neg

    loss_dev = weighted_std_deviation(loss,stats[:,0],np.sum(weights,axis=1),tot_pos+tot_neg)
    posrate_dev = weighted_std_deviation(posrate,stats[:,1],weights[:,0],tot_pos)
    negrate_dev = weighted_std_deviation(negrate,stats[:,2],weights[:,1],tot_neg)

    print("Validation stats:")
    print("Loss:",loss,"+/-",loss_dev,flush=True)
    print("Posrate:",posrate,"+/-",posrate_dev,flush=True)
    print("Negrate:",negrate,"+/-",negrate_dev,flush=True)
    print()
    
    datapoint = (youngests_age,loss,loss_dev,posrate,posrate_dev,negrate,negrate_dev)
    evaluated_models.add(youngests_age)
    datapoints.append(datapoint)

    datapoint_name = "{}/points/datapoint-{}.pt".format(sys.argv[3],datapoint[0])
    torch.save(datapoint,datapoint_name)
    
    plot_summary_and_report_best(datapoints)

  # finished outer loop

  # a final "cleanup"
  for p in my_processes:
    p.kill()
