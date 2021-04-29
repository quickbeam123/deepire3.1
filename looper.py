#!/usr/bin/env python3

# load inf_common before torch, so that torch is single threaded
import inf_common as IC

import torch
from torch import Tensor

import time

import pickle

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools,os,errno

from shutil import copyfile

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # To be called as in: ./looper.py run1.pkl run2.pkl ...
  #
  # Loading python 2 pickels in order, to determine which proofs to eventually take for the next loop training
  
  covered = set()

  solveds = []

  for i, pklname in enumerate(sys.argv[1:]):
    with open(pklname, 'rb') as f:
      solved = pickle.load(f, encoding="latin1") # loading a python2 pickle
      solved = set(solved)
      solveds.append(solved)
      
      print(i, pklname)
      
      if False and i > 6:
        solved_file_name = pklname.split("/")[-1]+".solved.txt"
        with open(solved_file_name,"w") as f:
          for probname in solved-covered:
            print(probname,file=f)
      
      if False and i > 6: # start looking at the logfiles to check for activation time limits for negative mining version
        source_folder_name = pklname[:-4]+"_s4k-on"
        for probname in solved-covered:
          logname = probname.replace("/","_") + ".log"
  
          full_logfile_path = source_folder_name+"/"+logname
          # print("Checking:",full_logfile_path)
          with open(full_logfile_path,'r') as g:
            for line in g:
              # print(line[:-1])
              if line.startswith("% Main loop iterations started:"):
                mis = int(line.split()[-1])
                
          #print("."+probname,"-al",mis)
          print(probname,"-al",int(mis*1.3))
      
      if False and i > 0: # a very hacky tool to copy just the contributed logs from where the pickles are to newly created folders under NEW_EXPORT, which must exist
        NEW_EXPORT = "smtlibbing/loop1dis10_10_plain/"
        
        source_folder_name = pklname[:-4]+"_s4k-on"
        print(source_folder_name)
        target_folder_name = NEW_EXPORT + source_folder_name.split("/")[-1]
        print(target_folder_name)
        
        try:
          os.mkdir(target_folder_name)
        except OSError as exc:
          if exc.errno != errno.EEXIST:
            raise
          pass
        
        # copy just the exciting ones:
        for probname in solved-covered:
          logname = probname.replace("/","_") + ".log"
          
          print(logname)
          
          copyfile(source_folder_name+"/"+logname, target_folder_name+"/"+logname)
      
      prev_size = len(covered)
      covered = covered | solved

      print("solved",len(solved),"added",len(covered)-prev_size,"total",len(covered))
      
