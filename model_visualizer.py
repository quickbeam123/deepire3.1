#!/usr/bin/env python3

import inf_common as IC

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools,math

import numpy as np

from multiprocessing import Pool

import matplotlib.cm as cm

from matplotlib import pyplot as plt

# from graphviz import Digraph # comment out for now

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

def contribute(repr,depth,posval,negval,val,logit):
  # print(repr,depth,isgood,val,logit)

  (abs_repr,sines) = repr
  sines = tuple(sines)

  group = abs_repr_groups[abs_repr]
  if sines in group:
    (models_val,models_logit,pos_labels,neg_labels) = group[sines]
    assert models_val == val
    assert models_logit == logit
  else:
    models_val = val
    models_logit = logit
    pos_labels = 0.0
    neg_labels = 0.0

  pos_labels += posval
  neg_labels += negval

  group[sines] = (models_val,models_logit,pos_labels,neg_labels)

  '''
  print(abs_repr)
  for sines, (models_val,models_logit,pos_labels,neg_labels) in group.items():
    print(sines,(models_val,models_logit,pos_labels,neg_labels) )
  print()
  '''

def eval_one(init,deriv,pars,pos_vals,negvals):
  for id, (thax,sine) in init:
    if thax == -1:
      st = "-1"
      abs_repr = "conj"
    elif thax in thax_to_str:
      st = thax_to_str[thax]
      abs_repr = st
    else:
      assert thax == 0
      st = str(thax)
      abs_repr = "other"
    
    repr = (abs_repr,[sine])
    
    # communication via st and sine
    getattr(model,"new_init")(id,[-1,-1,-1,-1,-1,sine],st)

    logit = model(id) # calling forward
    val = (logit >= 0.0) # interpreting the logit
    
    reprs[id] = repr
    
    depth = 1
    if abs_repr not in seen:
      seen.add(abs_repr)
      depths[abs_repr] = depth
    
    if pos_vals[id] + negvals[id] > 0.0:
      contribute(repr,depth,pos_vals[id],negvals[id],val,logit)
    
  for id, (rule) in deriv:
    if any((p not in reprs) for p in pars[id]):
      continue
  
    sines = [s for p in pars[id] for s in reprs[p][1]]
  
    if len(sines) > 1:
      continue
  
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
    
    abs_repr = f"{repr}({','.join([reprs[p][0] for p in pars[id]])})"

    repr = (abs_repr,sines)

    reprs[id] = repr

    if abs_repr not in seen:
      seen.add(abs_repr)
      
      depth = 1+max([depths[reprs[p][0]] for p in pars[id]])
      depths[abs_repr] = depth
    else:
      depth = depths[abs_repr]
    
    if pos_vals[id] + negvals[id] > 0.0:
      contribute(repr,depth,pos_vals[id],negvals[id],val,logit)

def get_logits(model,init,deriv,pars,selec,good,axioms):
  logits = {} # {id -> logit}
  
  for id, (thax,sine) in init:
    if thax == -1:
      st = "-1"
    elif id in axioms:
      st = axioms[id]
    else:
      assert thax == 0 or len(axioms)==0
      st = str(thax)

    # print(id,"has sine",sine)

    getattr(model,"new_init")(id,[-1,-1,-1,-1,-1,sine],st)
    logit = model(id) # calling forward
    
    logits[id] = logit
    
  for id, (rule) in deriv:
    if rule == 666:
      my_pars = pars[id]
      assert(len(my_pars) == 1)
      getattr(model,"new_avat")(id,[-1,-1,-1,my_pars[0]])
    else:
      getattr(model,"new_deriv{}".format(rule))(id,[-1,-1,-1,-1,rule],pars[id])

    logit = model(id) # calling forward
    logits[id] = logit

  return logits

def logit2color(l):
  prob = 1 / (1 + math.exp(-l))
  if prob >= 0.5:
    m = (1.0-prob)/4
    res = "{h:} 1 1".format(h=m)
  else:
    m = (0.625-prob/4)
    res = "{h:} 1 1".format(h=m)
  # print(l,"-->",prob,"-->",res)
  return res

def logits_by_size(model_file_name,init,deriv,pars,selec,good,axioms,res = defaultdict(list)):
  model = torch.jit.load(model_file_name)

  sizes = {} # id -> size
  
  for id, (thax,sine) in init:
    if thax == -1:
      st = "-1"
    elif id in axioms:
      st = axioms[id]
    else:
      assert thax == 0 or len(axioms)==0
      st = str(thax)

    # print(id,"has sine",sine)

    getattr(model,"new_init")(id,[-1,-1,-1,-1,-1,sine],st)
    logit = model(id) # calling forward
    
    size = 0
    sizes[id] = size
    res[size].append(logit)
  
  for id, (rule) in deriv:
    if rule == 666:
      my_pars = pars[id]
      assert(len(my_pars) == 1)
      getattr(model,"new_avat")(id,[-1,-1,-1,my_pars[0]])
    else:
      getattr(model,"new_deriv{}".format(rule))(id,[-1,-1,-1,-1,rule],pars[id])

    logit = model(id) # calling forward

    size = 1 + sum([sizes[p] for p in pars[id]])
    sizes[id] = size
    res[size].append(logit)

  return res

def par_logits_to_concl_logit_by_rule(task):
  i,model_file_name,prob = task
  (init,deriv,pars,selec,good,axioms) = prob

  res = defaultdict(list) # { rule -> [((parent's_logits),child_logit)]}

  model = torch.jit.load(model_file_name)
  
  for id, (thax,sine) in init:
    if thax == -1:
      st = "-1"
    elif id in axioms:
      st = axioms[id]
    else:
      assert thax == 0 or len(axioms)==0
      st = str(thax)

    # print(id,"has sine",sine)

    getattr(model,"new_init")(id,[-1,-1,-1,-1,-1,sine],st)
    
    # will be called if needed later (and cached)
    # logit = model(id) # calling forward
  
  for id, (rule) in deriv:
    if rule == 666:
      my_pars = pars[id]
      assert(len(my_pars) == 1)
      getattr(model,"new_avat")(id,[-1,-1,-1,my_pars[0]])
    else:
      getattr(model,"new_deriv{}".format(rule))(id,[-1,-1,-1,-1,rule],pars[id])

    logit = model(id) # calling forward

    res[rule].append((tuple([model(p) for p in pars[id]]),logit))

  print("Done",i)
  return res

def collect_rule_tuples(model_file_name,prob_data_list):
  # map
  tasks = [(i,model_file_name,prob) for i,(meta,prob) in enumerate(prob_data_list)]
  pool = Pool(processes=30) # number of cores to use
  results = pool.map(par_logits_to_concl_logit_by_rule, tasks, chunksize = 10)
  pool.close()
  pool.join()
  del pool

  big_res = defaultdict(list) # { rule -> [((parent's_logits),child_logit)]}
  # reduce
  for res in results:
    for rule,lst in res.items():
      big_res[rule] += lst

  return big_res

def plot_res(big_res,rule):
  fig, ax1 = plt.subplots(figsize=(5, 5))

  ax1.plot([(x[0][0]+x[0][1])/2 for x in big_res[rule]],[x[1] for x in big_res[rule]],".")
  ax1.set_xlim([-12,12])
  ax1.set_ylim([-12,12])

  plt.savefig(f"pokus_{rule}.png",dpi=250)
  plt.close()

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Load a torchscript model and
  # 1) either a name of log file (with a single proof) to parse
  # 2) or data_sign.pt / raw_log_data_avF_thaxVampThax.pt
  # 3) or a piece file or already processed (and abstraction-compressed) derivations (and whatever else is needed here)
  # visualize how the model evaluates the seen clauses and compare to ground truth
  #
  # Will also need to know whether sine levels / conj bit are present
  # and whether thax are really theory axioms from vampire or fixed axiom set as for mizar
  #
  # To be called as in:
  # 1) ./model_visualizer.py strat1new_better/l0_thax1000/data_sign.pt strat1new_better/raw_models/check-epoch60.pt strat1new_better/l0_thax1000/

  # Read rule names and thax
  rule_names = {}
  with open("inferences.info.txt","r") as f:
    i = 0
    for line in f:
      rule_names[i] = line[:-1]
      i += 1
  rule_names[666] = "AVATAR"

  # load a signature
  thax_sign,sine_sign,deriv_arits,thax_to_str = torch.load(sys.argv[1])
  print("Loaded data signature")

  ax_num_lookup = {} # name -> thax
  for thax,name in thax_to_str.items():
    if thax in thax_sign: # otherwise we decided we "replace this with I_unknown" (i.e., here: thax > 1000)
      ax_num_lookup[name] = thax

  '''
  prob_data_list = torch.load(sys.argv[3])

  if True:
    for (metainfo,(init,deriv,pars,selec,good,axioms)) in prob_data_list:
      metaprinted = False
      for id, (thax,sine) in init:
        if id in selec and thax == 699: # d1_funct_2
          if not metaprinted:
            print(metainfo)
            metaprinted = True
          print(id, thax, sine, id in good)

    exit(0)
  '''

  (_epoch,parts,_optim) = torch.load(sys.argv[2])
  # make sure we are in the eval mode
  for part in parts:
    part.eval()
    for param in part.parameters():
      param.requires_grad = False
  (init_embeds,sine_embellisher,deriv_mlps,eval_net) = parts

  # print(thax_sign)
  # print()
  # print(sine_sign)
  # print()
  # print(deriv_arits)
  for rule,arit in deriv_arits.items():
    print(rule,arit,rule_names[rule])
  print()

  logs = []
  with open("strat1new_better/loop0_logs.txt") as f:
    for line in f:
      logs.append(line[:-1])

  bydepth_pos = defaultdict(int) # "I saw a positively classified clause at depth d"
  bydepth_all = defaultdict(int) # "I saw a                       clause at depth d"

  bysize_truepos = defaultdict(int)
  bysize_pos = defaultdict(int)
  bysize_trueneg = defaultdict(int)
  bysize_neg = defaultdict(int)

  for i,log_file_name in enumerate(logs):
    print(i,"/",len(logs),log_file_name)

    result = IC.load_one(log_file_name)
    if result:
      probdata,time_elapsed = result
    else:
      print("Could not load supplied log file")

    (init,deriv,pars,selec,good,axioms) = probdata
    
    print(flush=True)

    reprs  = {} # abs_repr = "axiom(sine)" / "rule(abs_ids of premises)" -> abs_id

    abs_ids = {} # id -> abs_id

    depths = {} # abs_id -> depth
    sizes  = {} # abs_id -> size

    embeds = {} # abs_id -> tensor

    depth = 1
    size = 0
    for id, (thax,sine) in init:
      if thax == 0: # otherwise, it's -1, the conjecture
        ax_name = axioms[id]
        if ax_name in ax_num_lookup:
          thax = ax_num_lookup[ax_name]
        # otherwise stays 0, that's fine 0 means "I_unknown"
    
      if thax == -1:
        repr = "conj"
      elif thax == 0:
        repr = "other"
      else:
        repr = ax_name

      repr = "{}({})".format(repr,sine)

      if repr in reprs:
        abs_id = reprs[repr]
      else:
        abs_id = len(reprs)
        reprs[repr] = abs_id

      abs_ids[id] = abs_id

      depths[abs_id] = depth
      sizes[abs_id] = size

      # communication via st and sine
      init_embed = init_embeds[str(thax)].weight
      t = sine_embellisher(sine,init_embed)
      logit = eval_net(t)[0].item()

      # print(id,thax,sine,logit)

      embeds[abs_id] = t

      if logit >= 0.0:
        bydepth_pos[depth] += 1
      bydepth_all[depth] += 1
      
      if id in selec: # a relevant example
        if id in good:
          # print("seen a pos", id, size)
          bysize_pos[size] += 1
          if logit >= 0.0:
            bysize_truepos[size] += 1
        else:
          # print("seen a neg", id, size)
          bysize_neg[size] += 1
          if logit < 0.0:
            bysize_trueneg[size] += 1
      
    for id, (rule) in deriv:
      repr = "{}({})".format(rule_names[rule],",".join([str(abs_ids[p]) for p in pars[id]]))
      
      if repr in reprs:
        abs_id = reprs[repr]
      else:
        abs_id = len(reprs)
        reprs[repr] = abs_id

      abs_ids[id] = abs_id

      depth = 1+max(depths[abs_ids[p]] for p in pars[id])
      depths[abs_id] = depth
      
      size = 1+sum(sizes[abs_ids[p]] for p in pars[id])
      sizes[abs_id] = size
      
      par_embeds = [embeds[abs_ids[p]] for p in pars[id]]
      
      t = deriv_mlps[str(rule)](par_embeds)
      logit = eval_net(t)[0].item()

      embeds[abs_id] = t

      if logit >= 0.0:
        bydepth_pos[depth] += 1
      bydepth_all[depth] += 1
      
      if id in selec: # a relevant example
        if id in good:
          # print("seen a pos", id, size)
          bysize_pos[size] += 1
          if logit >= 0.0:
            bysize_truepos[size] += 1
        else:
          # print("seen a neg", id, size)
          bysize_neg[size] += 1
          if logit < 0.0:
            bysize_trueneg[size] += 1

    print("By size TRP/TNR",flush=True)
    # for s in range(1+max(max(bysize_pos,default=0),max(bysize_neg,default=0)),0,-1):
    for s in range(50):
      print('{: 3}'.format(s))
      if s in bysize_pos:
        print('   pos {:1.3f} (from {}/{})'.format(bysize_truepos[s]/bysize_pos[s],bysize_truepos[s],bysize_pos[s]))
      if s in bysize_neg:
        print('   neg {:1.3f} (from {}/{})'.format(bysize_trueneg[s]/bysize_neg[s],bysize_trueneg[s],bysize_neg[s]))
    print()
      
  print("By depth")
  for d,val in sorted(bydepth_all.items(),reverse=True):
    print('{: 3} {:1.3f} from {}/{}'.format(d,bydepth_pos[d]/val,bydepth_pos[d],val))
  print()

  exit(0)

  # print(thax_to_str)

  train_idx = torch.load("{}/training_index.pt".format(sys.argv[3]))
  valid_idx = torch.load("{}/validation_index.pt".format(sys.argv[3]))

  if False:
    (_,piece_name) = train_idx[0]
    data = torch.load("{}/pieces/{}".format(sys.argv[3],piece_name))
    (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) = data
  else:
    d_pn_pw = ("",0.0) # dummy_probname_and_probweight

    prob_data_list = [(d_pn_pw,torch.load("{}/pieces/{}".format(sys.argv[3],piece_name))) for (_,piece_name) in train_idx[0:30]]
    (joint_probname,joint_probweight),(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) = IC.compress_prob_data(prob_data_list)

  # probably doing something again; now just with raw model rather than the jit one

  max_depth_pos = 0

  transition_boolhist = defaultdict(int)

  embeds = {} # id -> tensor
  reprs  = {} # id -> representation of the abstract deriv tree
  records = [] # (logit,repr) # to be sorted, printed, and marveled at

  # TODO - try: vdepths = {} # id -> vdepth # `vampire depth' = simplifications (and avatar) don't increase depth, but rather inherit from the main premise?
  depths = {}  # id -> depth
  sizes = {}   # id -> size

  bydepth_pos = defaultdict(int) # "I saw a positively classified clause at depth d"
  bydepth_all = defaultdict(int) # "I saw a                       clause at depth d"

  for id, (thax,sine) in init:
    if thax == -1:
      repr = "conj"
    elif thax in thax_to_str:
      repr = thax_to_str[thax]
    else:
      assert thax == 0
      repr = "other"
    
    repr = "{}({})".format(repr,sine)
    reprs[id] = repr
    depths[id] = 1
    sizes[id] = 1

    # communication via st and sine
    init_embed = init_embeds[str(thax)].weight
    t = sine_embellisher(sine,init_embed)
    logit = eval_net(t)[0].item()

    transition_boolhist[((),(logit>=0.0))] += 1

    embeds[id] = t

    weight = pos_vals[id] + neg_vals[id]
    if weight > 0.0:
      if logit >= 0.0:
        bydepth_pos[1] += 1
      bydepth_all[1] += 1
    
      records.append((logit,pos_vals[id]/weight,1,1,repr))

  for id, (rule) in deriv:
    repr = "{}({})".format(rule_names[rule],",".join([reprs[p] for p in pars[id]]))
    reprs[id] = repr
    
    # print(id,rule)
    # print([depths[p] for p in pars[id]])
    # print([sizes[p] for p in pars[id]])
    
    depth = 1+max(depths[p] for p in pars[id])
    size = 1+sum(sizes[p] for p in pars[id])
    
    depths[id] = depth
    sizes[id] = size
    
    par_embeds = [embeds[p] for p in pars[id]]
    
    # print(id,rule,par_embeds)
    
    t = deriv_mlps[str(rule)](par_embeds)
    logit = eval_net(t)[0].item()

    if len(par_embeds) <= 2: # the left-out rest is urr only
      input = tuple(eval_net(t)[0].item() >= 0.0 for t in par_embeds)
      output = (logit>=0.0)
      '''
      if output and not all(input):
        print("Uphill!",rule_names[rule])
        print(input,output)
        print(repr)
        print()
      '''
      
      transition_boolhist[(input,output)] += 1

    embeds[id] = t

    weight = pos_vals[id] + neg_vals[id]
    if weight > 0.0:
      if logit >= 0.0:
        bydepth_pos[depth] += 1
      
        if depth > max_depth_pos:
          max_depth_pos = depth
          print("Improves max_depth_pos to",max_depth_pos)
          print(repr)
          print()
      
      bydepth_all[depth] += 1
    
      records.append((logit,pos_vals[id]/weight,depth,size,repr))

  records.sort()

  for logit,percent,depth,size,repr in records:
    print(logit,percent,depth,size,repr)

  print()
  for b,val in sorted(transition_boolhist.items()):
    print(val, b)

  print()
  print("By depth")
  for d,val in sorted(bydepth_all.items()):
    print('{: 3} {:1.3f} from {}/{}'.format(d,bydepth_pos[d]/val,bydepth_pos[d],val))
    
  exit(0)

  # 0-ary
  total = 10000
  hist = defaultdict(int)
  for i in range(total):
    t1 = torch.normal(0, 10, size=(128,))
    # tn = deriv_mlps["42"]((t1,))

    t1t = eval_net(t1)[0].item() > 0
    # tnt = eval_net(tn)[0].item() > 0

    hist[(t1t)] += 1

  for h,val in sorted(hist.items()):
    print(h,val/total)

  exit(0)

  # unary
  hist = defaultdict(int)
  for i in range(10000):
    t1 = torch.normal(0, 1, size=(128,))
    tn = deriv_mlps["42"]((t1,))

    t1t = eval_net(t1)[0].item() > 0
    tnt = eval_net(tn)[0].item() > 0

    hist[(t1t,tnt)] += 1

  for h,val in hist.items():
    print(h,val)

  exit(0)

  # binary
  hist = defaultdict(int)
  for i in range(10000):
    t1 = torch.normal(0, 1, size=(128,))
    t2 = torch.normal(0, 1, size=(128,))
    tn = deriv_mlps["44"]((t1,t2))

    t1t = eval_net(t1)[0].item() > 0
    t2t = eval_net(t2)[0].item() > 0
    tnt = eval_net(tn)[0].item() > 0

    hist[(t1t,t2t,tnt)] += 1

  for h,val in hist.items():
    print(h,val)

  exit(0)

  total = 64
  ts = [torch.normal(0, 1, size=(128,)) for _ in range(total) ]
  slots = [ [] for _ in range(total)]
  step = 1
  while len(ts) >= 1:
    idx = 0
    new_ts = []
    last_t = None
    for i,t in enumerate(ts):
      logit = eval_net(t)[0].item()
      
      slots[idx].append(logit)
      idx += step
      
      if last_t is not None:
        tn = deriv_mlps["44"]((last_t,t))
        new_ts.append(tn)
        last_t = None
      else:
        last_t = t

    ts = new_ts
    step *= 2

  for what_to_print in slots:
    for x in what_to_print:
      print(x,end=" ")
    print()

  '''
  print("start",norm,logit)
  for _ in range(30):
    tn = deriv_mlps["40"]((t,t))
    logit = eval_net(tn)[0].item()
    dist = torch.norm(tn-t).item()
    print("  resjump",dist,logit)
    t = tn
  '''

  exit(0)

  if True: # how the initial clauses (axioms) change values based on the sine level
    id = 0
    sines = sorted(list(sine_sign))
    # g = Digraph('G', filename='input_axioms.gv',format='png')

    lastTrueLevelHist = defaultdict(int)

    init_vals = [] # a list of (val,ax,la)

    for ax in range(1002):
      ax = ax-1 # starting from -1
      if ax == -1:
        st = "-1"
        la = "conj"
      elif ax == 0:
        st = "0"
        la = "input"
      else:
        st = thax_to_str[ax]
        la = st

      print("axiom",ax,la)
      init_embed = init_embeds[str(ax)].weight

      lastTrueLevel = -1
      prevSineLevel = -1
      for sine in sines:
        t = sine_embellisher(sine,init_embed)
        norm = torch.norm(t).item()
        logit = eval_net(t)[0].item()
        print("s:",sine,norm,logit)
        
        init_vals.append((logit,sine,ax,la,t))
        
        if logit >= 0.0: # still true
          assert(lastTrueLevel == prevSineLevel) # would get violated if we go from False to True again
          lastTrueLevel = sine
        
        # g.node(str(id),label="{} ({})".format(la,sine),style="filled",color=logit2color(logit))
        # if sine > 0:
        #   g.edge(str(id-1),str(id))
        id += 1

        prevSineLevel = sine

        # let's play an avatar game with the t
        for _ in range(100):
          tn = deriv_mlps["42"]((t,))
          logit = eval_net(tn)[0].item()
          dist = torch.norm(tn-t).item()
          print("  avjump",dist,logit)
          t = tn

      # print()

      lastTrueLevelHist[lastTrueLevel] += 1

    # g.render()

    for level,cnt in sorted(lastTrueLevelHist.items()):
      print(level,cnt)
    print()

    '''
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    keys = list(lastTrueLevelHist.keys())
    print(keys)

    keys.sort()
    vals = [lastTrueLevelHist[key] for key in keys]

    ax.pie(vals, labels = keys,autopct='%1.2f%%', startangle = 90)# , labeldistance=1.15 )

    plt.savefig("pie.png")
    '''

    init_vals.sort()
    for (val,sine,ax,la,t) in init_vals[-1000:]:
      if sine == 1:
        print (val,sine,ax,la)

    def chart(t_min,t_max,la_min,la_max):
      Z = []
      for y in np.arange(-1, 1, 0.01):
        Zy = []
        for x in np.arange(-1, 1, 0.01):
          z = eval_net(x*t_min+y*t_max)[0].item()
          Zy.append(z)
        Z.append(Zy)

      Z = np.array(Z)

      print("vmax",abs(Z).max(),"vmin",-abs(Z).max())

      fig, ax = plt.subplots()
      im = ax.imshow(Z, interpolation='nearest', cmap=cm.Spectral,
                 origin='lower', extent=[-1, 1, -1, 1],
      #            vmax=10, vmin=-10)
                vmax=abs(Z).max(), vmin=-abs(Z).max())
      fig.colorbar(im)

      plt.savefig("chart_{}_{}.png".format(la_min,la_max))
      plt.close()

    '''
    for i in range(10):
      (val,sine,ax,la,t) = init_vals[1+13*i]
      t_min = t
      la_min = la
      
      (val,sine,ax,la,t) = init_vals[-1-14*i]
      t_max = t
      la_max = la

      chart(t_min,t_max,la_min,la_max)
    '''

    '''
    for i in range(100):
      t = torch.normal(0, 1, size=(128,))
      
      print(i,torch.norm(t).item())
    
      chart(t,torch.normal(0, 1, size=(128,)),"r","r{}".format(i))
    '''

    exit(0)

  # load a log file
  result = IC.load_one(sys.argv[3])
  if result:
    probdata,time_elapsed = result
  else:
    print("Could not load supplied log file")
    exit(0)

  (init,deriv,pars,selec,good,axioms) = probdata

  # compute logits for all ids in the log
  logits = get_logits(model,init,deriv,pars,selec,good,axioms)

  # compute the depths; as we like to know
  depths = {} # id -> depth
  nodes_by_depth = defaultdict(list)

  for id, (thax,sine) in init:
    depths[id] = 0
    nodes_by_depth[0].append(id)

  for id, (rule) in deriv:
    d = 1+max( [depths[p] for p in pars[id]])
    depths[id] = d
    nodes_by_depth[d].append(id)

  random_last = None
  i = 0
  while len(nodes_by_depth[i]) > 0:
    print(i,len(nodes_by_depth[i]))
    random_last = random.choice(nodes_by_depth[i])
    i += 1

  print()

  # standard formatter
  def format_node(isSelec,isGood,logit):
    c = logit2color(logit)
    if isSelec:
      if isGood:
        return dict(shape='doubleoctagon',style="filled",color=c)
      else:
        return dict(shape='box',style="filled",color=c)
    else:
      return dict(shape = 'plaintext',fontcolor=c)

  '''
  # highlight mistakes formatter
  def format_node(isSelec,isGood,logit):
    if isSelec:
      if isGood:
        if logit < 0.0:
          return dict(shape='doubleoctagon',style="filled",color="blue")
        else:
          return dict(shape='doubleoctagon')
      else:
        if logit > 0.0:
          return dict(shape='box',style="filled",color="red")
        else:
          return dict(shape='box')
    else:
      return dict(shape = 'plaintext',fontcolor="black")
  '''

  # for plotting only one of the longest paths
  filter = lambda id : True
  if False:
    toBeSeen = {random_last}
    print("starting with",random_last)
    while random_last in pars:
      par_max_depth = -1
      max_depth_pars = None
      for p in pars[random_last]:
        toBeSeen.add(p)
        if depths[p] > par_max_depth:
          par_max_depth = depths[p]
          max_depth_pars = [p]
        elif depths[p] == par_max_depth:
          max_depth_pars.append(p)

      print("had pars",pars[random_last])
      print("par_max_depth",par_max_depth,"max_depth_pars:",max_depth_pars)
      
      random_last = random.choice(max_depth_pars)

      print("continue with",random_last)

    filter = lambda id : (id in toBeSeen)

  # plot the whole derivation
  if True:
    g = Digraph('G', filename='modviz.gv',format='png')
    # print(axioms)

    with g.subgraph() as s:
      s.attr(rank='same')
      for id, (thax,sine) in init:
        if not filter(id):
          continue
        
        if thax == -1:
          st = "CONJ"
        elif id in axioms:
          st = axioms[id]
        else:
          assert thax == 0 or len(axioms)==0
          st = rule_names[thax]
      
        s.node(str(id),label="{}: {} ({})".format(id,st,sine),**format_node(id in selec,id in good,logits[id]))

    MAX_WIDTH = 30
    rules = {}
    for id, rule in deriv:
      rules[id] = rule
    
    i = 1
    while len(nodes_by_depth[i]) > 0:
      l = sorted(nodes_by_depth[i])
      groups = [ l[MAX_WIDTH*i:MAX_WIDTH*(i+1)] for i in range(1+(len(l)-1)//MAX_WIDTH)]
      
      # print(l)
      
      print(i,"have groups",len(l),len(groups))
      
      for group in groups:
        print(i,"a group",len(group))
      
        with g.subgraph() as s:
          s.attr(rank='same')

          for id in group:
            if not filter(id):
              continue
          
            s.node(str(id),label="{}: {}".format(id,rule_names[rules[id]]),**format_node(id in selec,id in good,logits[id]))
              
            for p in pars[id]:
              if not filter(p):
                continue
                
              g.edge(str(p),str(id))
      i += 1
    
    # u = g.unflatten(stagger=3)
    g.attr(rankdir='LR')
    g.render()

    exit(0)


  exit(0)

  thax_sign,sine_sign,deriv_arits,thax_to_str = torch.load("{}/data_sign.pt".format(sys.argv[1]))
  print("Loaded data signature")
  train_data_idx = torch.load("{}/training_index.pt".format(sys.argv[1]))
  print("Loaded train data:",len(train_data_idx))
  valid_data_idx = torch.load("{}/validation_index.pt".format(sys.argv[1]))
  print("Loaded valid data:",len(valid_data_idx))
  data_idx = train_data_idx + valid_data_idx
  
  model = torch.jit.load(sys.argv[2]) # always load a new model -- it contains the lookup tables for the particular model
  
  seen = set() # what the repr already printed?
  depths = {} # reprs -> its term depth
  
  abs_repr_groups = defaultdict(dict) # abs_repr -> (sines -> (model_s_val,logic,pos_labels,neg_labels))

  for (size,probname) in data_idx:
    print("Opening",probname,"of size",size)
    data = torch.load("{}/pieces/{}".format(sys.argv[1],probname))
    (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) = data
    print("has",len(init),"init,",len(deriv),"deriv, and weight",tot_pos,"/",tot_neg)
    
    reprs = {} # id -> clause_string_representation
    eval_one(init,deriv,pars,pos_vals,neg_vals)

  '''
  for (depth,val),examples in sorted(hist.items(),reverse=True):
    print((depth,val),len(examples),examples[:10] if depth <= 3 else "")
  '''
  print()
  for abs_repr, group in abs_repr_groups.items():
    print(abs_repr)
    for sines, (models_val,models_logit,pos_labels,neg_labels) in group.items():
      print(sines,(models_val,models_logit,pos_labels > 0.0,neg_labels > 0.0) )
    print()
 
