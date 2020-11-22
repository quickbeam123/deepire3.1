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

from multiprocessing import Pool

'''
enum class InferenceRule : unsigned char {
  INPUT,
  GENERIC_FORMULA_TRANSFORMATION,
  NEGATED_CONJECTURE,
  ANSWER_LITERAL,
  CLAIM_DEFINITION,
  RECTIFY,
  CLOSURE,
  FLATTEN,
  ENNF,
  NNF,
  REDUCE_FALSE_TRUE,    ----> 10

  DEFINITION_FOLDING,
  THEORY_NORMALIZATION,
  SKOLEMIZE,
  CLAUSIFY,
  INTERNAL_FORMULA_TRANSFORMATION_LAST,
  GENERIC_SIMPLIFYING_INFERNCE,
  REORDER_LITERALS,
  REMOVE_DUPLICATE_LITERALS,
  TRIVIAL_INEQUALITY_REMOVAL,
  EQUALITY_RESOLUTION_WITH_DELETION,  -----> 20
  
  SUBSUMPTION_RESOLUTION,
  FORWARD_DEMODULATION,
  BACKWARD_DEMODULATION,
  FORWARD_SUBSUMPTION_DEMODULATION,
  BACKWARD_SUBSUMPTION_DEMODULATION,
  FORWARD_LITERAL_REWRITING,
  INNER_REWRITING,
  CONDENSATION,
  EVALUATION,
  INTERPRETED_SIMPLIFICATION,     ------> 30

  THEORY_FLATTENING,
  TERM_ALGEBRA_DISTINCTNESS,
  TERM_ALGEBRA_INJECTIVITY_SIMPLIFYING,
  HYPER_SUPERPOSITION_SIMPLIFYING,
  GLOBAL_SUBSUMPTION,
  DISTINCT_EQUALITY_REMOVAL,
  GAUSSIAN_VARIABLE_ELIMINIATION,
  INTERNAL_SIMPLIFYING_INFERNCE_LAST,
  GENERIC_GENERATING_INFERNCE,
  RESOLUTION,                    -------> 40
  
  CONSTRAINED_RESOLUTION,
  FACTORING,
  CONSTRAINED_FACTORING,
  SUPERPOSITION,
  CONSTRAINED_SUPERPOSITION,
  EQUALITY_FACTORING,
  EQUALITY_RESOLUTION,
  EXTENSIONALITY_RESOLUTION,
  TERM_ALGEBRA_INJECTIVITY_GENERATING,
  TERM_ALGEBRA_ACYCLICITY,      -------> 50
  
  
  /** Replaces a literal of the form C[s] with C[true] \/ s = false, where s is a boolean non-variable term */
  FOOL_PARAMODULATION,
  /** unit resulting resolution */
  UNIT_RESULTING_RESOLUTION,
  /** hyper-superposition */
  HYPER_SUPERPOSITION_GENERATING,
  /** generated as instance of its parent */
  INSTANCE_GENERATION, // used by InstGen. Fun fact: the inference has one parent (logically) but the age is set from two parents (and +1)!
  /* Instantiation */
  INSTANTIATION, // used for theory reasoning
  /** the last generating inference marker --
        inferences between GENERIC_GENERATING_INFERNCE and INTERNAL_GENERATING_INFERNCE_LAST will be automatically understood generating
        (see also isGeneratingInferenceRule) */
  INTERNAL_GENERATING_INFERNCE_LAST,


  /** equality proxy replacement */
  EQUALITY_PROXY_REPLACEMENT,
  /** definition of the equality proxy predicate in the form E(x,y) <=> x=y */
  EQUALITY_PROXY_AXIOM1,
  /** equality proxy axioms such as E(x,x) or ~E(x,y) \/ x=y */
  EQUALITY_PROXY_AXIOM2,
  /** unfolding by definitions f(x1,...,xn)=t */
  DEFINITION_UNFOLDING,

  /** introduction of new name p, p <=> C */
  PREDICATE_DEFINITION,
  /** unfolding predicate definitions */
  PREDICATE_DEFINITION_UNFOLDING,
  /** merging predicate definitions */
  PREDICATE_DEFINITION_MERGING,


  /** unused predicate definition removal */
  UNUSED_PREDICATE_DEFINITION_REMOVAL,
  /** pure predicate removal */
  PURE_PREDICATE_REMOVAL,
  /** inequality splitting */
  INEQUALITY_SPLITTING,
  /** inequality splitting name introduction */
  INEQUALITY_SPLITTING_NAME_INTRODUCTION,
  /** grounding */
  GROUNDING,
  /** equality axiom */
  EQUALITY_AXIOM,
  /** distinctness axiom */
  DISTINCTNESS_AXIOM,
  /** Introduction of formula to convert formulas used as argument positions.
   *  Such formulas have the form F->f(x)=1 or ~F->f(x)=0 */
  BOOLEAN_TERM_ENCODING,
  /** Elimination of FOOL expressions that makes a formula not syntactically first-order */
  FOOL_ELIMINATION,
  /** Elimination of $ite expressions */
  FOOL_ITE_ELIMINATION,
  /** Elimination of $let expressions */
  FOOL_LET_ELIMINATION,
  /** result of general splitting */
  GENERAL_SPLITTING,
  /** component introduced by general splitting */
  GENERAL_SPLITTING_COMPONENT,
  /** replacing colored constants by skolem functions */
  COLOR_UNBLOCKING,

  /** refutation in the SAT solver for InstGen */
  SAT_INSTGEN_REFUTATION,

  /** definition introduced by AVATAR */
  AVATAR_DEFINITION,
  /** component introduced by AVATAR */
  AVATAR_COMPONENT,
  /** refutation of a AVATAR splitting branch */
  AVATAR_REFUTATION,
  /** sat clause representing FO clause for AVATAR */
  AVATAR_SPLIT_CLAUSE,
  /** sat clause representing FO clause for AVATAR */
  AVATAR_CONTRADICTION_CLAUSE,
  /** sat color elimination */
  SAT_COLOR_ELIMINATION,
  /** obtain a formula from a clause */
  FORMULIFY,

  /** inference coming from outside of Vampire */
  EXTERNAL,

  /* FMB flattening */
  FMB_FLATTENING,
  /* Functional definition for FMB */
  FMB_FUNC_DEF,
  /* Definition Introduction for FMB */
  FMB_DEF_INTRO,
  /* Finite model not found */
  MODEL_NOT_FOUND,

  /* Adding sort predicate */
  ADD_SORT_PREDICATES,
  /* Adding sort functions */
  ADD_SORT_FUNCTIONS,

  /** a premise to skolemization */
  CHOICE_AXIOM,

  /* Induction hypothesis*/
  INDUCTION_AXIOM,
  /* Generalized nduction hypothesis*/
  GEN_INDUCTION_AXIOM,

  /* the unit clause against which the Answer is extracted in the last step */
  ANSWER_LITERAL_RESOLVER,

  /** A (first-order) tautology generated on behalf of a decision procedure,
   * whose propositional counterpart becomes a conflict clause in a sat solver */
  THEORY_TAUTOLOGY_SAT_CONFLICT,

  /** a not further specified theory axiom internally added by the class TheoryAxioms. */
  GENERIC_THEORY_AXIOM, // CAREFUL: adding rules here influences the theory_split_queue heuristic
  /** Some specific groups of axioms coming from TheoryAxioms.cpp" */
  THA_COMMUTATIVITY,
  THA_ASSOCIATIVITY,
  THA_RIGHT_IDENTINTY,
  THA_LEFT_IDENTINTY,
  THA_INVERSE_OP_OP_INVERSES,
  THA_INVERSE_OP_UNIT,
  THA_INVERSE_ASSOC,
  THA_NONREFLEX,
  THA_TRANSITIVITY,
  THA_ORDER_TOTALALITY,
  THA_ORDER_MONOTONICITY,
  THA_PLUS_ONE_GREATER,
  THA_ORDER_PLUS_ONE_DICHOTOMY,
  THA_MINUS_MINUS_X,
  THA_TIMES_ZERO,
  THA_DISTRIBUTIVITY,
  THA_DIVISIBILITY,
  THA_MODULO_MULTIPLY,
  THA_MODULO_POSITIVE,
  THA_MODULO_SMALL,
  THA_DIVIDES_MULTIPLY,
  THA_NONDIVIDES_SKOLEM,
  THA_ABS_EQUALS,
  THA_ABS_MINUS_EQUALS,
  THA_QUOTIENT_NON_ZERO,
  THA_QUOTIENT_MULTIPLY,
  THA_EXTRA_INTEGER_ORDERING,
  THA_FLOOR_SMALL,
  THA_FLOOR_BIG,
  THA_CEILING_BIG,
  THA_CEILING_SMALL,
  THA_TRUNC1,
  THA_TRUNC2,
  THA_TRUNC3,
  THA_TRUNC4,
  THA_ARRAY_EXTENSIONALITY,
  THA_BOOLEAN_ARRAY_EXTENSIONALITY, // currently applied to a formula, so won't propagate to clause->isTheoryAxiom()
  THA_BOOLEAN_ARRAY_WRITE1, // currently applied to a formula, so won't propagate to clause->isTheoryAxiom()
  THA_BOOLEAN_ARRAY_WRITE2, // currently applied to a formula, so won't propagate to clause->isTheoryAxiom()
  THA_ARRAY_WRITE1,
  THA_ARRAY_WRITE2,
  /** acyclicity axiom for term algebras */
  TERM_ALGEBRA_ACYCLICITY_AXIOM,
  TERM_ALGEBRA_DIRECT_SUBTERMS_AXIOM,
  TERM_ALGEBRA_SUBTERMS_TRANSITIVE_AXIOM,
  /** discrimination axiom for term algebras */
  TERM_ALGEBRA_DISCRIMINATION_AXIOM,
  /** distinctness axiom for term algebras */
  TERM_ALGEBRA_DISTINCTNESS_AXIOM,
  /** exhaustiveness axiom (or domain closure axiom) for term algebras */
  TERM_ALGEBRA_EXHAUSTIVENESS_AXIOM, // currently (sometimes) applied to a formula, so won't propagate to clause->isTheoryAxiom()
  /** exhaustiveness axiom (or domain closure axiom) for term algebras */
  TERM_ALGEBRA_INJECTIVITY_AXIOM,
  /** one of two axioms of FOOL (distinct constants or finite domain) */
  FOOL_AXIOM_TRUE_NEQ_FALSE,
  FOOL_AXIOM_ALL_IS_TRUE_OR_FALSE,
  /** the last internal theory axiom marker --
    axioms between THEORY_AXIOM and INTERNAL_THEORY_AXIOM_LAST will be automatically making their respective clauses isTheoryAxiom() true */
  INTERNAL_THEORY_AXIOM_LAST,
  /** a theory axiom which is not generated internally in Vampire */
  EXTERNAL_THEORY_AXIOM
}; // class InferenceRule
'''

def eval_one(init,deriv,pars,pos_vals,negvals):
  for id, (thax,sine) in init:
    if thax == -1:
      st = "-1"
      repr = "conj"
    elif thax in thax_to_str:
      st = thax_to_str[thax]
      repr = st
    else:
      assert thax == 0
      st = str(thax)
      repr = "other"
    
    repr = f"{repr}({sine})"
    
    # communication via st and sine
    getattr(model,"new_init")(id,[-1,-1,-1,-1,-1,sine],st)

    logit = model(id) # calling forward
    val = (logit >= 0.0) # interpreting the logit
    
    reprs[id] = repr
    
    if repr not in depths:
      depths[repr] = 0
      print(0,val,logit,pos_vals[id],negvals[id],repr)
      
      hist[(0,val)].append(repr)

  for id, (rule) in deriv:
    if rule == 666:
      my_pars = pars[id]
      assert(len(my_pars) == 1)
      getattr(model,"new_avat")(id,[-1,-1,-1,my_pars[0]])
      repr = "avat"
    else:
      getattr(model,"new_deriv{}".format(rule))(id,[-1,-1,-1,-1,rule],pars[id])
      repr = f"rule_{rule}"

    logit = model(id) # calling forward
    val = (logit >= 0.0) # interpreting the logit
    
    repr = f"{repr}({','.join([reprs[p] for p in pars[id]])})"
    reprs[id] = repr

    if repr not in depths:
      depth = 1+max([depths[reprs[p]] for p in pars[id]])
      depths[repr] = depth
      
      hist[(depth,val)].append(repr)
      
      if depth < 5:
        print(depth,val,logit,pos_vals[id],negvals[id],repr)

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Load a torchscript model and a set of logs, passed in a file as the final argument,
  # test the model on the logs (as if vampire was running) and report individual and average pos/neg rates
  #
  # To be called as in: ./model_visualizer.py folder_withraw_training/validation_index/and_with_data_sign.pt torch_script_model.pt

  thax_sign,sine_sign,deriv_arits,thax_to_str = torch.load("{}/data_sign.pt".format(sys.argv[1]))
  print("Loaded data signature")
  train_data_idx = torch.load("{}/training_index.pt".format(sys.argv[1]))
  print("Loaded train data:",len(train_data_idx))
  valid_data_idx = torch.load("{}/validation_index.pt".format(sys.argv[1]))
  print("Loaded valid data:",len(valid_data_idx))
  data_idx = train_data_idx + valid_data_idx
  
  model = torch.jit.load(sys.argv[2]) # always load a new model -- it contains the lookup tables for the particular model
  
  depths = {} # reprs -> its term depth (also works as "seen" set)

  hist = defaultdict(list) # (depth,val) -> examples

  for (size,probname) in data_idx[:100]:
    print("Opening",probname,"of size",size)
    data = torch.load("{}/pieces/{}".format(sys.argv[1],probname))
    (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) = data
    print("has",len(init),"init,",len(deriv),"deriv, and weight",tot_pos,"/",tot_neg)
    
    reprs = {} # id -> clause_string_representation
    eval_one(init,deriv,pars,pos_vals,neg_vals)

  print()

  for (depth,val),examples in sorted(hist.items(),reverse=True):
    print((depth,val),len(examples),examples[:10] if depth <= 3 else "")
 