#!/usr/bin/env python3

# a module of concepts common to the inference based model development

import torch

# multi_inf_paralels_config:

SCRATCH = "/raid/scratch/sudamar2"
MAX_EPOCH = 100

# DATA PREPARATION PARAMS:

TreatAvatarEmpties_JUSTFINAL = 1
TreatAvatarEmpties_INCLUDEALL = 2

def TreatAvatarEmptiesName(val):
  if val == TreatAvatarEmpties_JUSTFINAL:
    return "F"
  elif val == TreatAvatarEmpties_INCLUDEALL:
    return "E"

AVATAR_EMPTIES = TreatAvatarEmpties_JUSTFINAL

ThaxSource_THAX_FEATURE = 1
ThaxSource_AXIOM_NAMES = 2

def ThaxSourceName(val):
  if val == ThaxSource_THAX_FEATURE:
    return "VampThax"
  elif val == ThaxSource_AXIOM_NAMES:
    return "AxiomNames"

THAX_SOURCE = ThaxSource_THAX_FEATURE

SPLIT_AT_ACTIVATION = False
ONLY_GENERATING_PARENTS = False

# only take the first MAX_USED_AXIOM_CNT thax values to create embeddings for (all other will join 0)
# this needs to be done before/during the compression phase
# note that log-loading already introduced the axioms in the order of decreasing estimated usefulness
# only makes sense for THAX_SOURCE = ThaxSource_AXIOM_NAMES
MAX_USED_AXIOM_CNT = 3000

COMPRESSION_THRESHOLD = 20000

VALID_SPLIT_RATIO = 0.9

# these are now ignored in multi_inf_parallel_files_continuous.py
WHAT_IS_BIG = 12000
WHAT_IS_HUGE = 120000

# used for both training and model export (should be kept the same)
USE_SINE = True

# any other value than -1 (which means "off") will get hardwired during export into the model
FAKE_CONST_SINE_LEVEL = -1

# MODEL PARAMS:

# a hyper-parameter of the future model
EMBED_SIZE = 64

NonLinKind_TANH = 1
NonLinKind_RELU = 2

def NonLinKindName(val):
  if val == NonLinKind_TANH:
    return "Tanh"
  elif val == NonLinKind_RELU:
    return "ReLU"

NONLIN = NonLinKind_RELU

BOTTLENECK_EXPANSION_RATIO = 2 # is used halved for the eval layer (and sine layer?)

LAYER_NORM = True

# These two should probably used exclusively,
# also they are only imporant when we have (NONLIN == NonLinKind_RELU && LAYER_NORM == False)
CLIP_GRAD_NORM = None # either None of the max_norm value to pass to clip_grad_norm_
CLIP_GRAD_VAL = None  # either None of the clip_value value to pass to clip_grad_value_

DROPOUT = 0.3

# LEARNING PARAMS:

NUMPROCESSES = 20

TestRiskRegimen_VALIDATE = 1
TestRiskRegimen_OVERFIT = 2

def TestRiskRegimenName(val):
  if val == TestRiskRegimen_VALIDATE:
    return "VALIDATE"
  elif val == TestRiskRegimen_OVERFIT:
    return "OVERFIT"

TRR = TestRiskRegimen_VALIDATE

SWAPOUT = 0.0
LEARN_RATE = 0.00005
MOMENTUM = 0.9 # only for SGD

NON_CONSTANT_10_50_250_LR = True

# Corresponds to L2 regularization
WEIGHT_DECAY = 0.0

Optimizer_SGD = 1
Optimizer_ADAM = 2
Optimizer_ADAMW = 3

def OptimizerName(val):
  if val == Optimizer_SGD:
    return "SGD"
  elif val == Optimizer_ADAM:
    return "Adam"
  elif val == Optimizer_ADAMW:
    return "AdamW"

OPTIMIZER = Optimizer_ADAM

POS_WEIGHT_EXTRA = 1.0

FRACTIONAL_CHECKPOINTING = 5 # 0 means disabled, 1 does not make sense
