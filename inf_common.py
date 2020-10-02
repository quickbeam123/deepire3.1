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
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys,random

from enum import Enum

# MODEL PARAMS:

# a hyper-parameter of the future model
EMBED_SIZE = 60
NONLIN = torch.nn.Tanh() # torch.nn.ReLU()

class CatLayerKind(Enum):
  SMALL = 1
  BIGGER = 2  # as used at AITP
  DOUBLE_NONLIN = 3  # seems to make more sense

CAT_LAYER = CatLayerKind.BIGGER

class EvalLayerKind(Enum):
  LINEAR = 1
  NONLIN = 2

EVAL_LAYER = EvalLayerKind.NONLIN

class LayerNorm(Enum):
  OFF = 1
  ON = 2

LAYER_NORM = LayerNorm.OFF

# LEARNING PARAMS:

SWAPOUT = 0.0
LEARN_RATE = 0.001

POS_BIAS = 0.85
NEG_BIAS = 0.15

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
    if CAT_LAYER == CatLayerKind.SMALL:
      self.catter = torch.nn.Linear(arit*dim,dim)
    else: # for BIGGER and DOUBLE_NONLIN
      self.big = torch.nn.Linear(arit*dim,(arit+1)*dim//2)
      self.small = torch.nn.Linear((arit+1)*dim//2,dim)

  def forward(self,args : List[Tensor]) -> Tensor:
    x = torch.cat(args)
    if CAT_LAYER == CatLayerKind.SMALL:
      x = self.catter(x)
      x = NONLIN(x)
    else: # for BIGGER and DOUBLE_NONLIN
      x = self.big(x)
      x = NONLIN(x)
      x = self.small(x)

    if CAT_LAYER == CatLayerKind.DOUBLE_NONLIN:
      x = NONLIN(x)

    return x

def get_initial_model(init_hist,deriv_hist):
  init_embeds = torch.nn.ModuleDict()
  assert(-1 in init_hist) # to have conjecture embedding
  assert(0 in init_hist)  # to have user-fla embedding
  for i in init_hist:
    init_embeds[str(i)] = Embed(EMBED_SIZE)

  # to have the arity 1 and 2 defaults
  # NOTE: 1 and 2 don't conflict with proper rule indexes
  assert((1,1) in deriv_hist)
  assert((2,2) in deriv_hist)

  deriv_mlps = torch.nn.ModuleDict()
  for (rule,arit) in deriv_hist:
    deriv_mlps[str(rule)] = CatAndNonLinear(EMBED_SIZE,arit)
  
  if EVAL_LAYER == EvalLayerKind.LINEAR:
    eval_net = torch.nn.Linear(EMBED_SIZE,1)
  else:
    eval_net = torch.nn.Sequential(
         torch.nn.Linear(EMBED_SIZE,EMBED_SIZE//2),
         NONLIN,
         torch.nn.Linear(EMBED_SIZE//2,1))

  return torch.nn.ModuleList([init_embeds,deriv_mlps,eval_net])

def name_initial_model_suffix():
  return "_{}_{}_CatLay{}_EvalLay{}_LayerNorm{}.pt".format(EMBED_SIZE,str(NONLIN)[:4],CAT_LAYER.name,EVAL_LAYER.name,LAYER_NORM.name)

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
  
  # This is, how we envision inference:
  class InfRecNet(torch.nn.Module):
    store : Dict[int, Tensor] # each id stores its embedding
    
    def __init__(self,
        init_G : torch.nn.Module,
        init_0 : torch.nn.Module,'''

bigpart2 ='''        eval_net : torch.nn.Module):
      super(InfRecNet, self).__init__()

      self.store = {}
      
      self.init_G = init_G
      self.init_0 = init_0'''

bigpart_rec1='''
    @torch.jit.export
    def new_init{}(self, id: int, features : Tuple[int, int, int, int, int, int]) -> bool:
      embed = self.init_{}()
      self.store[id] = embed
      val = self.eval_net(embed)
      return val[0].item() >= 0.0'''

bigpart_rec2='''
    @torch.jit.export
    def new_deriv{}(self, id: int, features : Tuple[int, int, int, int, int], pars : List[int]) -> bool:
      rule = features[-1]
      arit = len(pars)
      par_embeds = [self.store[par] for par in pars]
      embed = self.deriv_{}(par_embeds)
      self.store[id] = embed
      val = self.eval_net(embed)
      return val[0].item() >= 0.0'''

bigpart3 = '''
  module = torch.jit.script(InfRecNet(
    init_embeds['-1'],
    init_embeds['0'],'''

bigpart4 = '''    eval_net
    ))
  module.save(name)'''

def create_saver(init_hist,deriv_hist):
  with open("inf_saver.py","w") as f:
    print(bigpart1,file=f)

    for i in sorted(init_hist):
      if i > 0: # the other ones are already there
        print("        init_{} : torch.nn.Module,".format(i),file=f)

    for (rule,arit) in sorted(deriv_hist):
      print("        deriv_{} : torch.nn.Module,".format(rule),file=f)

    print(bigpart2,file=f)

    for i in sorted(init_hist):
      if i > 0:
        print("      self.init_{} = init_{}".format(i,i),file=f)

    print("      self.deriv_1 = deriv_1",file=f)
    print("      self.deriv_2 = deriv_2",file=f)
    for (rule,arit) in sorted(deriv_hist):
      print("      self.deriv_{} = deriv_{}".format(rule,rule),file=f)
    print("      self.eval_net = eval_net",file=f)

    print(bigpart_rec1.format("G","G"),file=f)
    print(bigpart_rec1.format("0","0"),file=f)

    for i in sorted(init_hist):
      if i > 0:
        print(bigpart_rec1.format(str(i),str(i)),file=f)

    for (rule,arit) in sorted(deriv_hist):
      print(bigpart_rec2.format(str(rule),str(rule)),file=f)

    print(bigpart3,file=f)
    for i in sorted(init_hist):
      if i > 0: # the other ones are already there
        print("    init_embeds['{}'],".format(i),file=f)
    for (rule,arit) in sorted(deriv_hist):
      print("    deriv_mlps['{}'],".format(rule),file=f)
    print(bigpart4,file=f)

# Learning model class
class LearningModel(torch.nn.Module):
  def __init__(self,
      init_embeds : torch.nn.ModuleDict,
      deriv_mlps : torch.nn.ModuleDict,
      eval_net : torch.nn.Module,
      init,deriv,pars,selec,good):
    super(LearningModel,self).__init__()
  
    self.init_embeds = init_embeds
    self.deriv_mlps = deriv_mlps
    self.eval_net = eval_net
    
    self.criterion = torch.nn.BCEWithLogitsLoss()
  
    self.init = init
    self.deriv = deriv
    self.pars = pars
    self.selec = selec
    self.good = good
  
    self.pos_weight = POS_BIAS/len(good) if len(good) else 1.0
    self.neg_weight = NEG_BIAS/(len(selec)-len(good)) if (len(selec)-len(good)) else 1.0
  
  def contribute(self, id: int, embed : Tensor):
    val = self.eval_net(embed)
    
    if id in self.good:
      self.posTot += 1
      if val[0].item() >= 0.0:
        self.posOK += 1
      
      gold = torch.tensor([1.0])
      weight = self.pos_weight
    else:
      self.negTot += 1
      if val[0].item() < 0.0:
        self.negOK += 1
      
      gold = torch.tensor([0.0])
      weight = self.neg_weight

    return weight*self.criterion(val,gold)

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
      if SWAPOUT > 0.0 and random.random() < SWAPOUT:
        embed = self.init_embeds[str(0)]()
      else:
        embed = self.init_embeds[str(thax)]()
      
      store[id] = embed
      if id in self.selec: # was selected, will contribute to loss
        loss += self.contribute(id,embed)
    
    for id, rule in self.deriv:
      # print("deriv",id)
      
      par_embeds = [store[par] for par in self.pars[id]]
      
      if SWAPOUT > 0.0 and random.random() < SWAPOUT:
        arit = len(self.pars[id])
        embed = self.deriv_mlps[str(arit)](par_embeds)
      else:
        embed = self.deriv_mlps[str(rule)](par_embeds)
      
      store[id] = embed
      if id in self.selec: # was selected, will contribute to loss
        loss += self.contribute(id,embed)

    return (loss,self.posOK/self.posTot if self.posTot else 1.0,self.negOK/self.negTot if self.negTot else 1.0)

def get_ancestors(seed,pars):
  ancestors = set()
  todo = [seed]
  while todo:
    cur = todo.pop()
    if cur not in ancestors:
      ancestors.add(cur)
      if cur in pars:
        for par in pars[cur]:
          todo.append(par)
  return ancestors

def load_one(filename):
  print("Loading",filename)

  init : List[Tuple[int, Tuple[int, int, int, int, int, int]]] = []
  deriv : List[Tuple[int, Tuple[int, int, int, int, int]]] = []
  pars : Dict[int, List[int]] = {}
  selec = set()
  
  empty = None
  
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
        init.append((val[1],val[2:]))
      elif spl[0] == "d:":
        # d: [2,cl_id,age,weight,len,num_splits,rule,par1,par2,...]
        val = eval(spl[1])
        assert(val[0] == 2)
        deriv.append((val[1],tuple(val[2:7])))
        id = val[1]
        pars[id] = val[7:]
        
        update_depths(id,depths,max_depth)
        
      elif spl[0] == "a:":
        # a: [3,cl_id,age,weight,len,causal_parent or -1]
        # treat it as deriv (with one parent):
        val = eval(spl[1])
        assert(val[0] == 3)
        deriv.append((val[1],(val[2],val[3],val[4],1,666))) # 1 for num_splits, 666 for rule
        id = val[1]
        pars[id] = [val[-1]]
      
        update_depths(id,depths,max_depth)
      
      elif spl[0] == "s:":
        selec.add(int(spl[1]))
      elif spl[0] == "r:":
        pass # ingored for now
      elif spl[0] == "e:":
        empty = int(spl[1])
      elif spl[0] == "f:":
        # fake one more derived clause ("-1") into parents
        empty = -1
        pars[empty] = list(map(int,spl[1].split(",")))
        
        update_depths(id,depths,max_depth)
  
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

  good = get_ancestors(empty,pars)
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

  print("init: {}, deriv: {}, select: {}, good: {}".format(len(init),len(deriv),len(selec),len(good)))

  return (init,deriv,pars,selec,good)

def prepare_hists(prob_data):
  init_hist = defaultdict(int)
  deriv_hist = defaultdict(int)

  for probname, (init,deriv,pars,selec,good) in prob_data.items():
    for id, features in init:
      # print("init",id)
      goal = features[-3]
      thax = features[-2]
      
      # unite them together
      thax = -1 if features[-3] else features[-2]
      
      init_hist[thax] += 1

    # make sure we have 0 - the default embedding ...
    init_hist[0] += 1
    # ... and -1, the conjecture one (although no conjecture in smtlib!)
    init_hist[-1] += 1

    for id, features in deriv:
      rule = features[-1]
      arit = len(pars[id])

      deriv_hist[(rule,arit)] += 1 # rule always implies a fixed arity (hmm, maybe not for global_sumbsumption)

    # make sure we have arity 1 and 2 defaults
    deriv_hist[(1,1)] += 1
    deriv_hist[(2,2)] += 1

  return (init_hist,deriv_hist)

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
      thax = -1 if features[-3] else features[-2]
      new_init.append((id+clause_offset, thax))
    
      if id > id_max:
        id_max = id

    new_deriv = []
    for id, features in deriv:
      rule = features[-1]
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
