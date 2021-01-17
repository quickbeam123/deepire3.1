#!/usr/bin/env python3

import inf_common as IC

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

import numpy as np

import pickle

from multiprocessing import Pool

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Load a torchscript model and folder with piece indexes
  # test the model on the pieces and report individual and average pos/neg rates
  # finally, plot a logits graph
  #
  # To be called as in: ./analyze_tweaks.py mizar_gsd/thax2000/data_sign.pt mizar_gsd/thax2000/runX-selected/check-epoch587.pt <optional_subset>
  # 
  # <optional_subset> is stored in python2 pickle,
  # example location (on air05):
  # ~/deepire3.1/mizar_deepire3_d4869_10s_strat1nacc_model_loop1_runX_thax2000_d01_256_i587_nesqr-20.1_nesqc--1.0.pkl

  thax_sign,sine_sign,deriv_arits,thax_to_str,prob_name2id,prob_id2name = torch.load(sys.argv[1])
  print("Loaded data signature")
  
  (epoch,master_parts,optimizer) = torch.load(sys.argv[2])
  prob_tweaks = master_parts[0].getWeight()
  print("Loaded and unpacked prob_tweaks")
  
  subset = None
  if len(sys.argv) > 3:
    with open(sys.argv[3], 'rb') as f:
      subset = pickle.load(f, encoding="latin1") # loading a python2 pickle
      print("Loaded a subset filter of size",len(subset))

  if True:
    X = np.array(prob_tweaks)
    fig, ax = plt.subplots(figsize=(15,10))
    ax.plot(X[:,0],X[:,1],marker='.',markersize=1, color = "gray", linestyle="None",alpha=0.3)
    plt.savefig("tweaks_plain.png",dpi=250)
    plt.close(fig)
    print("Plotted into tweaks_plain.png")
    exit(0)

  X = np.array(prob_tweaks)
  fig, ax = plt.subplots(figsize=(15,10))

  if subset is None:
    clustering = KMeans(n_clusters=10, random_state=0).fit(X)
    # clustering = AffinityPropagation(random_state=5).fit(X) # does not converge
    # clustering = MeanShift(bandwidth=0.05).fit(X)
    
    # clustering = DBSCAN(eps=0.01, min_samples=20).fit(X) # was a bit weird
    
    print(clustering.cluster_centers_) # works for KMeans/AffinityPropagation/MeanShift
    
    # print(clustering.core_sample_indices_)
    # print(clustering.components_)
    
    labels = set(clustering.labels_)
    for label in labels:
      ax.plot(X[clustering.labels_ == label,0],X[clustering.labels_ == label,1],marker='.',markersize=1, linestyle="None")
  
    ax.plot(clustering.cluster_centers_[:,0],clustering.cluster_centers_[:,1],marker='o', linestyle="None")

  else:
    isInSubset = np.array([prob_id2name[i] in subset for i in range(len(X))])
  
    ax.plot(X[~isInSubset,0],X[~isInSubset,1],marker='.',markersize=1, color = "b", linestyle="None")
    ax.plot(X[isInSubset,0],X[isInSubset,1],marker='.',markersize=1, color = "r", linestyle="None")

  plt.savefig("tweaks.png",dpi=250)
  plt.close(fig)
  print("Plotted into tweaks.png")
  
 
