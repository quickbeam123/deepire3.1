#!/usr/bin/env python3

# a module of concepts common to the inference based model development

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
from torch import Tensor

torch.set_num_threads(1)

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys,random

import hyperparams as HP

class Embed(torch.nn.Module):
  weight: Tensor
  
  def __init__(self, dim : int):
    super(Embed, self).__init__()
    
    self.weight = torch.nn.parameter.Parameter(torch.Tensor(dim))
    self.reset_parameters()
  
  def reset_parameters(self):
    torch.nn.init.normal_(self.weight)

  def forward(self) -> Tensor:
    return self.weight

class CatAndNonLinear(torch.nn.Module):
  
  def __init__(self, dim : int, arit: int):
    super(CatAndNonLinear, self).__init__()
    
    if HP.DROPOUT > 0.0:
      self.prolog = torch.nn.Dropout(HP.DROPOUT)
    else:
      self.prolog = torch.nn.Identity(arit*dim)
    
    if HP.NONLIN == HP.NonLinKind_TANH:
      self.nonlin = torch.nn.Tanh()
    else:
      self.nonlin = torch.nn.ReLU()
    
    if HP.CAT_LAYER == HP.CatLayerKind_SMALL:
      self.first = torch.nn.Linear(arit*dim,dim)
    else: # for BIGGER and DOUBLE_NONLIN
      self.first = torch.nn.Linear(arit*dim,(arit+1)*dim//2)
      self.second = torch.nn.Linear((arit+1)*dim//2,dim)

    if HP.LAYER_NORM == HP.LayerNorm_ON:
      self.epilog = torch.nn.LayerNorm(dim)
    else:
      self.epilog = torch.nn.Identity(dim)

  def forward_impl(self,args : List[Tensor]) -> Tensor:
    x = torch.cat(args)
    
    x = self.prolog(x)
    
    if HP.CAT_LAYER == HP.CatLayerKind_SMALL:
      x = self.first(x)
      x = self.nonlin(x)
    else: # for BIGGER and DOUBLE_NONLIN
      x = self.first(x)
      x = self.nonlin(x)
      x = self.second(x)

    if HP.CAT_LAYER == HP.CatLayerKind_DOUBLE_NONLIN:
      x = self.nonlin(x)

    x = self.epilog(x)

    return x
    
  # this whole method is bogus, just to make torch.jit.script happy
  def forward(self,args : List[Tensor]) -> Tensor:
    return args[0]

class CatAndNonLinearBinary(CatAndNonLinear):
  def forward(self,args : List[Tensor]) -> Tensor:
    return self.forward_impl(args)

class CatAndNonLinearMultiary(CatAndNonLinear):
  def forward(self,args : List[Tensor]) -> Tensor:
    i = 0
    while True:
      pair = args[i:i+2]
      
      if len(pair) == 2:
        args.append(self.forward_impl(pair))
        i += 2
      else:
        assert(len(pair) == 1)
        return pair[0]
    return args[0] # this is bogus, just to make torch.jit.script happy

def get_initial_model(init_sign,deriv_arits):
  init_embeds = torch.nn.ModuleDict()
  assert(-1 in init_sign) # to have conjecture embedding
  assert(0 in init_sign)  # to have user-fla embedding
  for i in init_sign:
    init_embeds[str(i)] = Embed(HP.EMBED_SIZE)

  # to have the arity 1 and 2 defaults
  # NOTE: 1 and 2 don't conflict with proper rule indexes
  assert(deriv_arits[1] == 1)
  assert(deriv_arits[2] == 2)

  deriv_mlps = torch.nn.ModuleDict()
  for rule,arit in deriv_arits.items():
    if arit <= 2:
      deriv_mlps[str(rule)] = CatAndNonLinearBinary(HP.EMBED_SIZE,arit)
    else:
      assert(arit == 3)
      deriv_mlps[str(rule)] = CatAndNonLinearMultiary(HP.EMBED_SIZE,2) # binary tree builder
  
  if HP.EVAL_LAYER == HP.EvalLayerKind_LINEAR:
    eval_net = torch.nn.Sequential(
         torch.nn.Dropout(HP.DROPOUT) if HP.DROPOUT > 0.0 else torch.nn.Identity(HP.EMBED_SIZE),
         torch.nn.Linear(HP.EMBED_SIZE,1))
  else:
    eval_net = torch.nn.Sequential(
         torch.nn.Dropout(HP.DROPOUT) if HP.DROPOUT > 0.0 else torch.nn.Identity(HP.EMBED_SIZE),
         torch.nn.Linear(HP.EMBED_SIZE,HP.EMBED_SIZE//2),
         torch.nn.Tanh() if HP.NONLIN == HP.NonLinKind_TANH else torch.nn.ReLU(),
         torch.nn.Linear(HP.EMBED_SIZE//2,1))

  return torch.nn.ModuleList([init_embeds,deriv_mlps,eval_net])

def name_initial_model_suffix():
  return "_{}_{}_CatLay{}_EvalLay{}_LayerNorm{}_Dropout{}.pt".format(
    HP.EMBED_SIZE,
    HP.NonLinKindName(HP.NONLIN),
    HP.CatLayerKindName(HP.CAT_LAYER),
    HP.EvalLayerKindName(HP.EVAL_LAYER),
    HP.LayerNormName(HP.LAYER_NORM),
    HP.DROPOUT)

def name_learning_regime_suffix():
  return "_lr{}_p{}n{}_swapout{}_trr{}.txt".format(
    HP.LEARN_RATE,
    HP.POS_BIAS,
    HP.NEG_BIAS,
    HP.SWAPOUT,
    HP.TestRiskRegimenName(HP.TRR))

bigpart1 = '''#!/usr/bin/env python3

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys,random

def save_net(name,parts,parts_copies):
  for part,part_copy in zip(parts,parts_copies):
    part_copy.load_state_dict(part.state_dict())
    
    # eval mode and no gradient
    part_copy.eval()
    for param in part_copy.parameters():
      param.requires_grad = False

  # from here on only use the updated copies
  (init_embeds,deriv_mlps,eval_net) = parts_copies
  
  initEmbeds = {}
  for ax_name,embed in init_embeds.items():
    initEmbeds[ax_name] = embed.weight
  
  # This is, how we envision inference:
  class InfRecNet(torch.nn.Module):
    store : Dict[int, Tensor] # each id stores its embedding
    initEmbeds : Dict[str, Tensor]
    
    def __init__(self,
        initEmbeds : Dict[str, Tensor],'''

bigpart2 ='''        eval_net : torch.nn.Module):
      super(InfRecNet, self).__init__()

      self.store = {}
      
      self.initEmbeds = initEmbeds'''
      
bigpart_no_longer_rec1 = '''
    @torch.jit.export
    def forward(self, id: int) -> float:
      val = self.eval_net(self.store[id])
      return val[0].item()

    @torch.jit.export
    def new_init(self, id: int, features : Tuple[int, int, int, int, int, int], name: str) -> None:
      if name in self.initEmbeds:
        embed = self.initEmbeds[name]
      else:
        embed = self.initEmbeds["0"]
      self.store[id] = embed'''

bigpart_rec2='''
    @torch.jit.export
    def new_deriv{}(self, id: int, features : Tuple[int, int, int, int, int], pars : List[int]) -> None:
      rule = features[-1]
      arit = len(pars)
      par_embeds = [self.store[par] for par in pars]
      embed = self.deriv_{}(par_embeds)
      self.store[id] = embed'''

bigpart_avat = '''
    @torch.jit.export
    def new_avat(self, id: int, features : Tuple[int, int, int, int]) -> None:
      par = features[-1]
      par_embeds = [self.store[par]]
      embed = self.deriv_666(par_embeds) # special avatar code
      self.store[id] = embed'''

bigpart3 = '''
  module = torch.jit.script(InfRecNet(
    initEmbeds,'''

bigpart4 = '''    eval_net
    ))
  module.save(name)'''

def create_saver(init_sign,deriv_arits,thax_to_str):
  with open("inf_saver.py","w") as f:
    print(bigpart1,file=f)

    '''
    for i in sorted(init_sign):
      if i > 0: # the other ones are already there
        print("        init_{} : torch.nn.Module,".format(i),file=f)
    '''

    for rule in sorted(deriv_arits):
      print("        deriv_{} : torch.nn.Module,".format(rule),file=f)

    print(bigpart2,file=f)

    '''
    for i in sorted(init_sign):
      if i > 0:
        print("      self.init_{} = init_{}".format(i,i),file=f)
    '''

    print("      self.deriv_1 = deriv_1",file=f)
    print("      self.deriv_2 = deriv_2",file=f)
    for rule in sorted(deriv_arits):
      print("      self.deriv_{} = deriv_{}".format(rule,rule),file=f)
    print("      self.eval_net = eval_net",file=f)

    print(bigpart_no_longer_rec1,file=f)
    
    '''
    print(bigpart_rec1.format("0","0"),file=f)

    for i in sorted(init_sign):
      if i > 0:
        print(bigpart_rec1.format(thax_to_str[i] if i in thax_to_str else str(i),str(i)),file=f)
    '''

    for rule in sorted(deriv_arits):
      if rule < 666: # avatar done differently in bigpart3
        print(bigpart_rec2.format(str(rule),str(rule)),file=f)

    if 666 in deriv_arits:
      print(bigpart_avat,file=f)

    print(bigpart3,file=f)
    '''
    for i in sorted(init_sign):
      if i > 0: # the other ones are already there
        print("    init_embeds['{}'],".format(i),file=f)
    '''
    for rule in sorted(deriv_arits):
      print("    deriv_mlps['{}'],".format(rule),file=f)
    print(bigpart4,file=f)

# Learning model class
class LearningModel(torch.nn.Module):
  def __init__(self,
      init_embeds : torch.nn.ModuleDict,
      deriv_mlps : torch.nn.ModuleDict,
      eval_net : torch.nn.Module,
      init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg):
    super(LearningModel,self).__init__()
  
    self.init_embeds = init_embeds
    self.deriv_mlps = deriv_mlps
    self.eval_net = eval_net
    
    self.init = init
    self.deriv = deriv
    self.pars = pars
    self.pos_vals = pos_vals
    self.neg_vals = neg_vals
  
    pos_weight = HP.POS_BIAS/tot_pos if tot_pos > 0.0 else 1.0
    neg_weight = HP.NEG_BIAS/tot_neg if tot_neg > 0.0 else 1.0
    
    self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight/neg_weight]))
  
  def contribute(self, id: int, embed : Tensor):
    val = self.eval_net(embed)
    
    pos = self.pos_vals[id]
    neg = self.neg_vals[id]
    
    self.posTot += pos
    self.negTot += neg
    
    if val[0].item() >= 0.0:
      self.posOK += pos
    else:
      self.negOK += neg
    
    return (pos+neg)*self.criterion(val,torch.tensor([pos/(pos+neg)]))

  # Construct the whole graph and return its loss
  # TODO: can we keep the graph after an update?
  def forward(self) -> Tensor:
    store : Dict[int, Tensor] = {} # each id stores its embedding
    
    self.posOK = 0
    self.posTot = 0
    self.negOK = 0
    self.negTot = 0
    
    loss = torch.zeros(1)
    
    for id, thax in self.init:
      if HP.SWAPOUT > 0.0 and random.random() < HP.SWAPOUT:
        embed = self.init_embeds[str(0)]()
      else:
        embed = self.init_embeds[str(thax)]()
      
      store[id] = embed
      if id in self.pos_vals or id in self.neg_vals:
        loss += self.contribute(id,embed)
    
    for id, rule in self.deriv:
      # print("deriv",id)
      
      par_embeds = [store[par] for par in self.pars[id]]
      
      if HP.SWAPOUT > 0.0 and random.random() < HP.SWAPOUT:
        arit = len(self.pars[id])
        embed = self.deriv_mlps[str(arit)](par_embeds)
      else:
        embed = self.deriv_mlps[str(rule)](par_embeds)
      
      store[id] = embed
      if id in self.pos_vals or id in self.neg_vals:
        loss += self.contribute(id,embed)

    return (loss,self.posOK/self.posTot if self.posTot else 1.0,self.negOK/self.negTot if self.negTot else 1.0)

def get_ancestors(seed,pars,**kwargs):
  ancestors = kwargs.get("known_ancestors",set())
  # print("Got",len(ancestors))
  todo = [seed]
  while todo:
    cur = todo.pop()
    if cur not in ancestors:
      ancestors.add(cur)
      if cur in pars:
        for par in pars[cur]:
          todo.append(par)
  return ancestors

def abstract_initial(features):
  goal = features[-3]
  thax = features[-2]
  # unite them together
  thax = -1 if features[-3] else features[-2]
  return thax

def abstract_deriv(features):
  rule = features[-1]
  return rule

def load_one(filename,max_size = None):
  print("Loading",filename)

  init : List[Tuple[int, Tuple[int, int, int, int, int, int]]] = []
  deriv : List[Tuple[int, Tuple[int, int, int, int, int]]] = []
  pars : Dict[int, List[int]] = {}
  selec = set()
  
  axioms : Dict[int, str] = {}
  
  empty = None
  good = set()
  
  depths = defaultdict(int)
  max_depth = 0
  
  def update_depths(id,depths,max_depth):
    ps = pars[id]
    depth = max([depths[p] for p in ps])+1
    depths[id] = depth
    if depth > max_depth:
      max_depth = depth

  with open(filename,'r') as f:
    for line in f:
      if max_size and len(init)+len(deriv) > max_size:
        return None
      
      # print(line)
      if line.startswith("% Refutation found."):
        break
      if line.startswith("% # SZS output start Saturation."):
        print("Skipping. Is SAT.")
        return None
      spl = line.split()
      if spl[0] == "i:":
        val = eval(spl[1])
        assert(val[0] == 1)
        id = val[1]
        init.append((id,abstract_initial(val[2:])))
        
        goal = val[-3]
        
        if len(spl) > 2 and not goal: # axiom name reported and this is not a conjecture clause
          axioms[id] = spl[2]
          
      elif spl[0] == "d:":
        # d: [2,cl_id,age,weight,len,num_splits,rule,par1,par2,...]
        val = eval(spl[1])
        assert(val[0] == 2)
        deriv.append((val[1],abstract_deriv(tuple(val[2:7]))))
        id = val[1]
        pars[id] = val[7:]
        
        update_depths(id,depths,max_depth)
        
      elif spl[0] == "a:":
        # a: [3,cl_id,age,weight,len,causal_parent or -1]
        # treat it as deriv (with one parent):
        val = eval(spl[1])
        assert(val[0] == 3)
        deriv.append((val[1],abstract_deriv((val[2],val[3],val[4],1,666)))) # 1 for num_splits, 666 for rule
        id = val[1]
        pars[id] = [val[-1]]
      
        update_depths(id,depths,max_depth)
      
      elif spl[0] == "s:":
        selec.add(int(spl[1]))
      elif spl[0] == "r:":
        pass # ingored for now
      elif spl[0] == "e:":
        empty = int(spl[1])
        
        # THIS IS THE INCLUSIVE AVATAR STRATEGY; comment out if you only want those empties that really contributed to the final contradiction
        # good = good | get_ancestors(empty,pars,known_ancestors=good)
        
      elif spl[0] == "f:":
        # fake one more derived clause ("-1") into parents
        empty = -1
        pars[empty] = list(map(int,spl[1].split(",")))
        
        update_depths(empty,depths,max_depth)
          
  assert(empty is not None)

  # NOTE: there are some things that should/could be done differently in the future
  #
  # *) in non-discount saturation algorithms, not every selection which participates in
  #  a proof needs to be good one. If the said selected clause does not participate
  #  in a generating inference necessary for the proof, it could have stayed "passive"
  #
  # *) for AVATAR one could consider learning from all the empty clauses rather than just
  #  the ones connected to "f"; it would then also become relevant that clauses get
  #  "retracted" from active as the model changes, so a selection long time ago,
  #  that got retracted in the meantime anyway, should not be considered bad
  #  for the current empty clause
  #  Note, however, that when learning from more than one empty clause indepentently,
  #  we would start getting clauses which are both good and bad; if we were to call
  #  all of these good anyway (because we only ever want to err on "the other side")
  #  this whole considiration becomes moot

  # one more goodness-collecting run;
  # 1) for the sake of the "f"-empty clause
  # 2) if the corresponding line for the common "e"-empty is commented out, this will collect at least once
  good = good | get_ancestors(empty,pars,known_ancestors=good)
  good = good & selec # proof clauses that were never selected don't count

  # TODO: consider learning only from hard problems!
  
  # E.g., solveable by a stupid strategy (age-only), get filtered out
  if not selec or not good:
    print("Skipping, degenerate.")
    return None

  # Don't be afraid of depth!
  '''
  if max_depth > 100:
    print("Skipping, is too deep.")
    return None
  '''

  print("init: {}, deriv: {}, select: {}, good: {}, axioms: {}".format(len(init),len(deriv),len(selec),len(good),len(axioms)))

  return (init,deriv,pars,selec,good,axioms)

def prepare_signature(prob_data_list):
  init_sign = set()
  deriv_arits = {}
  axiom_hist = defaultdict(int)

  for probname, (init,deriv,pars,selec,good,axioms) in prob_data_list:
    for id, features in init:
      # already abstracted
      thax = features
      init_sign.add(thax)

    # make sure we have 0 - the default embedding ...
    init_sign.add(0)
    # ... and -1, the conjecture one (although no conjecture in smtlib!)
    init_sign.add(-1)

    for id, features in deriv:
      # already abstracted
      rule = features
      arit = len(pars[id])

      if arit > 2:
        deriv_arits[rule] = 3 # the multi-ary way
      elif rule in deriv_arits and deriv_arits[rule] != arit:
        deriv_arits[rule] = 3 # mixing 1 and 2?
      else:
        deriv_arits[rule] = arit

    # make sure we have arity 1 and 2 defaults
    deriv_arits[1] = 1
    deriv_arits[2] = 2
  
    for id,ax in axioms.items():
      axiom_hist[ax] += 1

  return (init_sign,deriv_arits,axiom_hist)

def axiom_names_instead_of_thax(init_sign,axiom_hist,prob_data_list):
  # (we didn't parse anything than 0 and -1 anyway:)
  assert(0 in init_sign and (len(init_sign) == 1 or len(init_sign) == 2 and -1 in init_sign))
  
  new_prob_data_list = []
  
  ax_idx = {}
  thax_to_str = {}
  good_ax_cnt = 0
  for ax,num in sorted(axiom_hist.items(),key = lambda x : x[1]):
    # print(ax,num)
    
    if num >= 100: # change this constant to get something reasonable
      good_ax_cnt += 1
      ax_idx[ax] = good_ax_cnt
      thax_to_str[good_ax_cnt] = ax
      # print(ax,"is",good_ax_cnt)

  for (probname,(init,deriv,pars,selec,good,axioms)) in prob_data_list:
    new_init = []
    for id, features in init:
      # already abstracted
      thax = features
      if thax == 0: # don't name the conjecture
        if id in axioms and axioms[id] in ax_idx:
          thax = ax_idx[axioms[id]]
      new_init.append((id,thax))
      init_sign.add(thax)

    new_prob_data_list.append((probname,(new_init,deriv,pars,selec,good,axioms)))

  return init_sign,new_prob_data_list,thax_to_str

def normalize_prob_data(prob_data):
  # 1) it's better to have them in a list (for random.choice)
  # 2) it's better to just keep the relevant information (more compact memory footprint)
  # 3) it's better to disambiguate clause indices, so that any union of problems will make sense as on big graph
  
  prob_data_list = []
  clause_offset = 0

  for probname, (init,deriv,pars,selec,good) in prob_data.items():
    id_max = 0
    new_init = []
    for id, features in init:
      # already abstracted
      thax = features
      new_init.append((id+clause_offset, thax))
    
      if id > id_max:
        id_max = id

    new_deriv = []
    for id, features in deriv:
      # already abstracted
      rule = features
      new_deriv.append((id+clause_offset, rule))
      
      if id > id_max:
        id_max = id

    new_pars = {}
    for id, ids_pars in pars.items():
      new_pars[id+clause_offset] = [par+clause_offset for par in ids_pars]

    new_selec = {id + clause_offset for id in selec}
    new_good = {id + clause_offset for id in good}

    prob_data_list.append((probname,(new_init,new_deriv,new_pars,new_selec,new_good)))

    clause_offset += (id_max+1)

  return prob_data_list

def compress_prob_data(some_probs):
  # Takes an iterable of "probname, (init,deriv,pars,selec,good)" and
  # "hashes them structurally" (modulo the features we care about) to just one graph to learn from
  # initially intended to be used with just a singleton some_probs
  # (possible to compress pairs and triples, etc, to obtain "larger minibatches")
  #
  # Should replace normalize_prob_data above, although that one processes the whole dataset
  # and the new one should be applied piece-by-piece
  #
  # The selected features are
  # *) "thax" for init (can be -1 signifying conjecture clause, 0 for regular input and thax_ids otherwise)
  # *) and "rule" for deriv (arity is implict in the number of parents

  id_cnt = 0
  out_probname = ""
  
  abs2new = {} # maps (thax/rule,par_new_ids) to new_id (the structurally hashed one)
  
  out_init = []
  out_deriv = []
  out_pars = {}
  out_pos_vals = defaultdict(float)
  out_neg_vals = defaultdict(float)
  out_tot_pos = 0.0
  out_tot_neg = 0.0

  for probname, (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg) in some_probs:
    # reset for evey problem in the list
    old2new = {} # maps old_id to new_id (this is the not-necessarily-injective map)
  
    if out_probname:
      out_probname += "+"+probname
    else:
      out_probname = probname

    for old_id, features in init:
      # already abstracted
      thax = features

      abskey = (thax)

      if abskey not in abs2new:
        new_id = id_cnt
        id_cnt += 1

        abs2new[abskey] = new_id
        out_init.append((new_id,thax))
      else:
        new_id = abs2new[abskey]

      old2new[old_id] = new_id

    for old_id, features in deriv:
      # already abstracted
      rule = features

      new_pars = [old2new[par] for par in pars[old_id]]
      
      abskey = tuple([rule]+new_pars)

      if abskey not in abs2new:
        new_id = id_cnt
        id_cnt += 1

        abs2new[abskey] = new_id
        
        out_deriv.append((new_id,rule))
        out_pars[new_id] = new_pars
      else:
        new_id = abs2new[abskey]

      old2new[old_id] = new_id

    for old_id,val in pos_vals.items():
      out_pos_vals[old2new[old_id]] += val
    for old_id,val in neg_vals.items():
      out_neg_vals[old2new[old_id]] += val

    out_tot_pos += tot_pos
    out_tot_neg += tot_neg

  print("Compressed to",len(out_init),len(out_deriv),len(out_pars),len(pos_vals),len(neg_vals),out_tot_pos,out_tot_neg)
  return out_probname, (out_init,out_deriv,out_pars,out_pos_vals,out_neg_vals,out_tot_pos,out_tot_neg)

def big_data_prob(prob_data_list):
  # compute the big graph union - currently unused - was too big to use at once
  
  # print(prob_data_list)

  big_init = list(itertools.chain.from_iterable(init for probname, (init,deriv,pars,selec,good) in prob_data_list ))
  big_deriv = list(itertools.chain.from_iterable(deriv for probname, (init,deriv,pars,selec,good) in prob_data_list ))
  big_pars = dict(ChainMap(*(pars for probname, (init,deriv,pars,selec,good) in prob_data_list )))
  big_selec = set(itertools.chain.from_iterable(selec for probname, (init,deriv,pars,selec,good) in prob_data_list ))
  big_good = set(itertools.chain.from_iterable(good for probname, (init,deriv,pars,selec,good) in prob_data_list ))

  # print(big_init)

  return (big_init,big_deriv,big_pars,big_selec,big_good)

import matplotlib.pyplot as plt

def plot_one(filename,times,train_losses,train_posrates,train_negrates,valid_losses,valid_posrates,valid_negrates):
  fig, ax1 = plt.subplots()
  
  color = 'tab:red'
  ax1.set_xlabel('time (epochs)')
  ax1.set_ylabel('loss', color=color)
  tl, = ax1.plot(times, train_losses, "--", linewidth = 1, label = "train_loss", color=color)
  vl, = ax1.plot(times, valid_losses, "-", linewidth = 1,label = "valid_loss", color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('pos/neg-rate', color=color)  # we already handled the x-label with ax1
  
  tpr, = ax2.plot(times, train_posrates, "--", label = "train_posrate", color = "blue")
  tnr, = ax2.plot(times, train_negrates, "--", label = "train_negrate", color = "cyan")
  vpr, = ax2.plot(times, valid_posrates, "-", label = "valid_posrate", color = "blue")
  vnr, = ax2.plot(times, valid_negrates, "-", label = "valid_negrate", color = "cyan")
  ax2.tick_params(axis='y', labelcolor=color)

  # For pos and neg rates, we know the meaningful range:
  ax2.set_ylim([-0.05,1.05])

  fig.tight_layout()  # otherwise the right y-label is slightly clipped

  plt.legend(handles = [tl,vl,tpr,tnr,vpr,vnr], loc='lower left') # loc = 'best' is rumored to be unpredictable
  
  plt.savefig(filename,dpi=250)
  plt.close(fig)
