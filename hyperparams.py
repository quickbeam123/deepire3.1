#!/usr/bin/env python3

# a module of concepts common to the inference based model development

import torch

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

AXCNT_CUTOFF = 10 # only makes sense for THAX_SOURCE = ThaxSource_AXIOM_NAMES

# MODEL PARAMS:

# a hyper-parameter of the future model
EMBED_SIZE = 128

NonLinKind_TANH = 1
NonLinKind_RELU = 2

def NonLinKindName(val):
  if val == NonLinKind_TANH:
    return "Tanh"
  elif val == NonLinKind_RELU:
    return "ReLU"

NONLIN = NonLinKind_RELU

CatLayerKind_SMALL = 1
CatLayerKind_BIGGER = 2  # as used at AITP
CatLayerKind_DOUBLE_NONLIN = 3  # seems to make more sense

def CatLayerKindName(val):
  if val == CatLayerKind_SMALL:
    return "SMALL"
  elif val == CatLayerKind_BIGGER:
    return "BIGGER"
  elif val == CatLayerKind_DOUBLE_NONLIN:
    return "DOUBLE_NONLIN"

CAT_LAYER = CatLayerKind_DOUBLE_NONLIN

EvalLayerKind_LINEAR = 1
EvalLayerKind_NONLIN = 2

def EvalLayerKindName(val):
  if val == EvalLayerKind_LINEAR:
    return "LINEAR"
  elif val == EvalLayerKind_NONLIN:
    return "NONLIN"

EVAL_LAYER = EvalLayerKind_NONLIN

LayerNorm_OFF = 1
LayerNorm_ON = 2

def LayerNormName(val):
  if val == LayerNorm_OFF:
    return "OFF"
  elif val == LayerNorm_ON:
    return "ON"

LAYER_NORM = LayerNorm_ON

DROPOUT = 0.5

DEEPER = True

# LEARNING PARAMS:

TestRiskRegimen_VALIDATE = 1
TestRiskRegimen_OVERFIT = 2

def TestRiskRegimenName(val):
  if val == TestRiskRegimen_VALIDATE:
    return "VALIDATE"
  elif val == TestRiskRegimen_OVERFIT:
    return "OVERFIT"

TRR = TestRiskRegimen_OVERFIT

SWAPOUT = 0.0
LEARN_RATE = 0.001

Optimizer_SGD = 1
Optimizer_ADAM = 2

def OptimizerName(val):
  if val == Optimizer_SGD:
    return "SGD"
  elif val == Optimizer_ADAM:
    return "ADAM"

OPTIMIZER = Optimizer_ADAM

POS_WEIGHT_EXTRA = 2.0
