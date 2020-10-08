#!/usr/bin/env python3

# a module of concepts common to the inference based model development

import torch

# MODEL PARAMS:

# a hyper-parameter of the future model
EMBED_SIZE = 12

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

DROPOUT = 0.0

# LEARNING PARAMS:

SWAPOUT = 0.0
LEARN_RATE = 0.0007

POS_BIAS = 0.9
NEG_BIAS = 0.1
