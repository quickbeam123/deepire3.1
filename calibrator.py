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

import matplotlib.pyplot as plt

# To release claimed memory back to os; Call:   libc.malloc_trim(ctypes.c_int(0))
import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

# models are global (anyway just read-only) to all workers (does not work on mac)

def eval_on_one(piece_name):
  data = torch.load("{}/pieces/{}".format(sys.argv[1],piece_name))
  
  with torch.no_grad():
    model = IC.EvalMultiModel(models,data)
    model.eval()
    (loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg) = model()

  return (loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg)

def worker(q_in, q_out):
  log = sys.stdout

  while True:
    (piece_name,isValidation) = q_in.get()
    
    start_time = time.time()
    datapoint = eval_on_one(piece_name)
    
    q_out.put((piece_name,isValidation,datapoint,start_time,time.time()))

    libc.malloc_trim(ctypes.c_int(0))

def save_checkpoint(epoch, model, optimizer):
  print("checkpoint",epoch)

  check_name = "{}/check-epoch{}.pt".format(sys.argv[2],epoch)
  check = (epoch,model,optimizer)
  torch.save(check,check_name)

def load_checkpoint(filename):
  return torch.load(filename)

def save_datapoint(datapoint,piece_name):
  print("Saving datapoint for",piece_name)

  datapoint_name = "{}/points/datapoint-{}".format(sys.argv[3],piece_name)
  torch.save(datapoint,datapoint_name)

def load_datapoint(filename):
  return torch.load(filename)

def weighted_std_deviations(weighted_means,scaled_values,weights,weight_sum):
  weights = np.expand_dims(weights,1)
  
  values = np.divide(scaled_values, weights, out=np.ones_like(scaled_values), where=weights!=0.0) # whenever the respective weight is 0.0, the result should be understood as 1.0
  
  squares = (values - weighted_means)**2
  std_devs = np.sqrt(np.sum(weights*squares,axis=0) / weight_sum)
  return std_devs

def plot_datapoint(piece_name,isValidation,datapoint):
  print("Plotting datapoint for",piece_name)

  (loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg) = datapoint

  losses = loss_sum/(tot_pos+tot_neg)
  posrates = posOK_sum/tot_pos if tot_pos > 0.0 else tf.ones(len(posOK_sum))
  negrates = negOK_sum/tot_neg if tot_neg > 0.0 else tf.ones(len(negOK_sum))
  
  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('time (epochs)')
  ax1.set_ylabel('loss', color=color)
  vl, = ax1.plot(models_nums, losses, "-", linewidth = 1,label = "loss", color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  # ax1.set_ylim([0.45,0.6])

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('pos/neg-rate', color=color)  # we already handled the x-label with ax1

  vpr, = ax2.plot(models_nums, posrates, "-", label = "posrate", color = "blue")
  vnr, = ax2.plot(models_nums, negrates, "-", label = "negrate", color = "cyan")
  ax2.tick_params(axis='y', labelcolor=color)

  # For pos and neg rates, we know the meaningful range:
  ax2.set_ylim([-0.05,1.05])

  fig.tight_layout()  # otherwise the right y-label is slightly clipped

  plt.legend(handles = [vl,vpr,vnr], loc='lower left') # loc = 'best' is rumored to be unpredictable

  plotname = "{}/plots/plot-{}-{}.png".format(sys.argv[3],piece_name.split(".")[0],"validation" if isValidation else "training")
  plt.savefig(plotname,dpi=250)
  plt.close(fig)

def add_datapoint(datapoint,piece_name,isValidation):
  print("Adding datapoint for",piece_name)
  
  (loss_sum,posOK_sum,negOK_sum,tot_pos,tot_neg) = datapoint
  
  loss_sums.append(np.array(loss_sum))
  posOK_sums.append(np.array(posOK_sum))
  negOK_sums.append(np.array(negOK_sum))
  tot_poss.append(tot_pos)
  tot_negs.append(tot_neg)
  validations.append(isValidation)

def plot_summary_and_report_best(summary_kind,loss_sums,posOK_sums,negOK_sums,tot_poss,tot_negs):
  print("plot_summary_and_report_best for",summary_kind)

  # changing meaning of tot_pos/tot_neg, I know
  tot_pos = np.sum(tot_poss)
  tot_neg = np.sum(tot_negs)
  
  if tot_pos == 0.0 or tot_neg == 0.0:
    print("Aborting, can't normalize yet!")
    return
  
  # CAREFULE: could divide by zero!
  losses = np.sum(loss_sums,axis=0)/(tot_pos+tot_neg)
  posrates = np.sum(posOK_sums,axis=0)/tot_pos
  negrates = np.sum(negOK_sums,axis=0)/tot_neg

  idx = np.nanargmin(losses)
  print("Best",summary_kind,"loss model:",models_nums[idx],end=" ")
  print("loss",losses[idx],"posrate",posrates[idx],"negrate",negrates[idx])

  '''
  print()
  for idx,loss in sorted(enumerate(losses),key=lambda x: -x[1]):
    print(models_nums[idx])
    print("loss",losses[idx],"posrate",posrates[idx],"negrate",negrates[idx])
  '''

  # Let's go to compute variances

  losses_devs = weighted_std_deviations(losses,loss_sums,tot_poss+tot_negs,tot_pos+tot_neg)
  posrates_devs = weighted_std_deviations(posrates,posOK_sums,tot_poss,tot_pos)
  negrates_devs = weighted_std_deviations(negrates,negOK_sums,tot_negs,tot_neg)

  # fake a text output (of the already read data) as it would come out of multi_inf_parallel_files(continuous)
  if summary_kind == "union":
    with open("{}/as_if_run_{}".format(sys.argv[3],IC.name_learning_regime_suffix()), 'w') as f:
      for idx,nominal_idx in enumerate(models_nums):
        print(f"Epoch {nominal_idx} finished at <fake_time>",file=f)
        print(f"Loss: {losses[idx]} +/- {losses_devs[idx]}",file=f)
        print(f"Posrate: {posrates[idx]} +/- {posrates_devs[idx]}",file=f)
        print(f"Negrate: {negrates[idx]} +/- {negrates_devs[idx]}",file=f)
        print(file=f)

  plotname = "{}/plot_{}.png".format(sys.argv[3],summary_kind)
  IC.plot_with_devs(plotname,models_nums,losses,losses_devs,posrates,posrates_devs,negrates,negrates_devs)

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Load a bunch of checkpoints and start properly evaluating them on train/valid/both
  #
  # To be called as in ./calibrator.py <folder1> <folder2> <folder3>
  #
  # Expects <folder1> to contain "training_data.pt" and "validation_data.pt"
  #
  # <folder2> to contain files "check-epoch<number>.pt"
  #
  # The log, the converging plots, are to be saved to <folder3>
  
  # Careful -- ANNOINGLY, THE EVALUATION DEPENDS on HP.POS_WEIGHT_EXTRA
  
  # global redirect of prints to the just open "logfile"
  log = open("{}/eval{}".format(sys.argv[3],IC.name_learning_regime_suffix()), 'w')
  sys.stdout = log
  sys.stderr = log
  
  # DATA PREPARATION
  
  train_data_idx = torch.load("{}/training_index.pt".format(sys.argv[1]))
  print("Loaded train data:",len(train_data_idx))
  valid_data_idx = torch.load("{}/validation_index.pt".format(sys.argv[1]))
  print("Loaded valid data:",len(valid_data_idx))
  
  feed_sequence = [(piece_name,False,size) for (size,piece_name) in train_data_idx] + [(piece_name,True,size) for (size,piece_name) in valid_data_idx]
  
  if True:
    # take the largest only later, i.e. by size, then name, decreasing
    feed_sequence.sort(key=lambda x: (x[2],x[0]), reverse=True) # this is not a numeric sort, but that's OK. We just want to interleave training and validation examples
    # I thought the above seems to be biased towards the easir proofs,
    # however, the better explanation was that the loss on training problems is generally better
    # during eval (as opposed to during training) as dropout is obviously turned off
    
    # even this is not perfect as running for longer on this eventually gives better posrates and worse negrates than initially
    # my guess is that this order considers the larger (thus harder) problems (generally) earlier and only later does the more uniform rest
    # feed_sequence.sort(reverse=True) # this is not a numeric sort, but that's OK. We just want to interleave training and validation examples
  else:
    pass # TODAY, we want to keep validation tasks in the back, so as to just validate!

  print(flush=True)

  # MODEL LOADING
  
  checkpoint_names = []
  for filename in os.listdir(sys.argv[2]):
    if filename.startswith("check-epoch") and filename.endswith(".pt"):
      checkpoint_names.append((int(filename[11:-3]),filename))
  checkpoint_names.sort()

  models = []
  models_nums = []
  for num,filename in checkpoint_names:

    # print(num,filename)
    (epoch,model,optimizer) = load_checkpoint(f"{sys.argv[2]}/{filename}")
    assert num == epoch

    models.append(model)
    models_nums.append(num)

  print(f"Loaded {len(models)} models",flush=True)

  # LOAD DATAPOINTS if ALREADY HAVING SOME

  loss_sums = []   # each element a tensor of size len(models)
  posOK_sums = []  # each element a tensor of size len(models)
  negOK_sums = []  # each element a tensor of size len(models)
  tot_poss = []    # each element a float
  tot_negs = []    # each element a float
  validations = [] # each element a bool

  for filename in os.listdir(f"{sys.argv[3]}/points"):
    if filename.startswith("datapoint-") and filename.endswith(".pt"):
      print("Found saved datapoint",filename)

      our_piece_name = filename[10:]
      datapoint = load_datapoint(f"{sys.argv[3]}/points/{filename}")

      found = False
      for i,(piece_name,isValidation,size) in enumerate(feed_sequence):
        if our_piece_name == piece_name:
          del feed_sequence[i]
          found = True
          break
      assert(found)
      # add to vectors
      add_datapoint(datapoint,piece_name,isValidation)

  # do one plotting of union before you start looping:

  plot_summary_and_report_best("union",
    np.array(loss_sums),
    np.array(posOK_sums),
    np.array(negOK_sums),
    np.array(tot_poss),
    np.array(tot_negs))

  # exit(0)

  # LOOPING

  print()
  print("Starting timer",flush=True)
  start_time = time.time()

  MAX_ACTIVE_TASKS = 5
  num_active_tasks = 0

  q_in = torch.multiprocessing.Queue()
  q_out = torch.multiprocessing.Queue()
  my_processes = []
  for i in range(MAX_ACTIVE_TASKS):
    p = torch.multiprocessing.Process(target=worker, args=(q_in,q_out))
    p.start()
    my_processes.append(p)

  while feed_sequence or num_active_tasks > 0:
    # we use parts_copies as a counter of idle children in the pool
    while num_active_tasks < MAX_ACTIVE_TASKS and feed_sequence:
      
      num_active_tasks += 1
      (piece_name,isValidation,size) = feed_sequence.pop()
      
      print(time.time() - start_time,"starting {} job on problem".format("validation" if isValidation else "training"),piece_name,"of size",size,flush=True)
      q_in.put((piece_name,isValidation))
      
      print()

    (piece_name,isValidation,datapoint,time_start,time_end) = q_out.get() # this may block
    
    num_active_tasks -= 1

    print(time.time() - start_time,"job finished at on problem",piece_name,"started",time_start-start_time,"finished",time_end-start_time,"took",time_end-time_start,flush=True)
    print()
    
    print("Recording for",piece_name,"isValidation",isValidation,flush=True)
    save_datapoint(datapoint,piece_name)

    # first, plot this datapoint:
    plot_datapoint(piece_name,isValidation,datapoint)
    
    # add to vectors
    add_datapoint(datapoint,piece_name,isValidation)

    # now refresh the summary plots (and report bests)
    
    plot_summary_and_report_best("union",
      np.array(loss_sums),
      np.array(posOK_sums),
      np.array(negOK_sums),
      np.array(tot_poss),
      np.array(tot_negs))

    mask_validation = np.array(validations)

    plot_summary_and_report_best("validation",
      np.array(loss_sums)[mask_validation],
      np.array(posOK_sums)[mask_validation],
      np.array(negOK_sums)[mask_validation],
      np.array(tot_poss)[mask_validation],
      np.array(tot_negs)[mask_validation])
    plot_summary_and_report_best("training",
      np.array(loss_sums)[~mask_validation],
      np.array(posOK_sums)[~mask_validation],
      np.array(negOK_sums)[~mask_validation],
      np.array(tot_poss)[~mask_validation],
      np.array(tot_negs)[~mask_validation])

  # a final "cleanup"
  for p in my_processes:
    p.kill()
