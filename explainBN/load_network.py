import zipfile
import os
import pandas as pd
import pathlib
import numpy as np
import json

from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination

# LOAD BAYESIAN NETWORK

def load_network(network_name, online = False, verbose=False):
  """ Download a network from the internet, convert it to a PGMPY Bayesian Network, 
      select a target node, attach verbal explanations if available
      Available networks: "asia", "cancer", "earthquake", "sachs", "survey", 
                          "alarm", "child", "barley", 
                          "child", "insurance", "mildew", "water", "hailfinder", 
                          "hepar2", "win95pts"
  """
  if online:
    url = f"https://www.bnlearn.com/bnrepository/{network_name}/{network_name}.bif.gz"
    os.system(f"wget {url} -q")
    fn = f"{network_name}.bif.gz"
    os.system(f"gzip -qd -f {fn} -q")
    fn = f"{network_name}.bif"
    reader = BIFReader(fn)
    os.system(f"rm {fn}")
  else:
    fn = pathlib.Path(__file__).parent
    fn /= f"../exampleBNs/{network_name}.bif"
    reader = BIFReader(fn)
    
  model = reader.get_model()
  model.states = reader.get_states()
  
  # Decorate the model
  model, target, evidence_nodes = \
    decorate_model(model, model_name = network_name)
  

  # Precompute baseline marginal distribution of target
  model.baselines = {}
  v = VariableElimination(model)
  for node in model.nodes:
    model.baselines[node] = v.query(variables=[node], 
                                    evidence={}, 
                                    show_progress=False)
    
  return model, target, evidence_nodes

def decorate_model(model, model_name=None):
   
  
  fn = pathlib.Path(__file__).parent
  fn /= f"../exampleBNs/{model_name}.json"
  with open(fn, mode='r') as file:
    explanation_json = json.load(file)
  try:
    pass
  except:
    explanation_json = {}
    
  print(explanation_json)
    
  try:
    target_node = explanation_json["target_node"]
  except KeyError:
    target_node = np.random.choice(model.nodes())
   
  try:
    target_state = explanation_json["target_state"]
  except KeyError:
    target_state = np.random.choice(model.states[target_node])
    
  target = (target_node, target_state)

  try:
    evidence_nodes = explanation_json["evidence_nodes"]
  except KeyError:
    evidence_nodes = list(model.nodes())
    evidence_nodes.remove(target[0])

  # Check validity of target and evidence nodes
  assert target[0] in model.nodes() and target[1] in model.states[target[0]]
  for evidence_node in evidence_nodes:
    assert evidence_node in model.nodes()

  # Add node descriptions
  try:
    model.variable_description = \
      pd.DataFrame([(node, explanation_json["nodes"][node]["description"]) 
                    for node in model.nodes], 
                    columns= ['Variable', 'Meaning'])
    
  except KeyError:
    model.variable_description = \
      pd.DataFrame([(node, node) for node in model.nodes], 
                   columns= ['Variable', 'Meaning'])
  
  # Add explanation of states
  model.explanation_dictionary = {}
  for node in model.nodes:
    for state in model.states[node]:
      try:
        d = explanation_json["nodes"][node]["states"][state]
      except KeyError:
        d = {}
      
      model.explanation_dictionary[(node, state)] = {}
      
      model.explanation_dictionary[(node, state)]["explanation"] = \
        d["explanation"] if "explanation" in d else f"{node} = {state}"
      
      model.explanation_dictionary[(node, state)]["contrastive_explanation"] = \
        d["contrastive_explanation"] \
        if "contrastive_explanation" in d \
        else f"not {node} = {state}"
      
      model.explanation_dictionary[(node, state)]["polarity"] = \
        d["polarity"] if "polarity" in d else "positive"
  
  return model, target, evidence_nodes
  

# READ DNE FILES
rhs = lambda s : re.match(r".*(.*) = (.*).*", s).group(2).strip(";")
my_tuple_reader = lambda s : list(filter(lambda s : len(s) > 0, 
                                         s.strip("()").replace(" ", "").split(",")))
find_floats = lambda s : list(map(float,re.findall(r'\d+(?:\.\d+)?', s)))

def read_dne(dne_file_location):
  # Read file content
  with open(dne_file_location, 'r') as file:
    lines = file.readlines()
  
  # Iterate and read the nodes
  node_names = {}
  node_states = {}
  raw_edges = []
  raw_probs = {}

  lines = iter(lines)
  for line in lines:
    if line.startswith("node"):
      current_node = line.split()[1]

    elif re.match(r".*states =.*", line):
      node_states[current_node] = my_tuple_reader(rhs(line))

    elif re.match(r".*parents =.*", line):
      parents = my_tuple_reader(rhs(line))
      for parent in parents:
        raw_edges.append((parent, current_node))
    
    elif re.match(r".*title =.*", line):
      node_names[current_node] = rhs(line).strip("\"")
    
    elif re.match(r".*probs =.*", line):
      prob_header = next(lines)
      m = re.match(r"[ \t]*\/\/([^\/\n]*)(\/\/.*)?", prob_header)
      states = re.split('\s+', m.group(1).strip())
      parents = m.group(2)
      if parents is not None:
        parents = parents.strip().strip(r"//").split()
      
      prob_line = next(lines)
      probs = []
      parent_states = [] if parents is not None else None
      while True:
        numbers = find_floats(prob_line)
        probs.append(numbers)
        if parents is not None:
          parent_states.append(re.match(r".*\/\/ (.*) ;?", prob_line).group(1).split())
        if re.match(r".*(\/\/)?.*;", prob_line):
          break
        else:
          prob_line = next(lines)
      probs = np.array(probs)
      raw_probs[current_node] = {"probs" : probs, 
                                 "states" : states, 
                                 "parent_aliases" : parents,
                                 "parent_states" : parent_states}

  # Process conditional probability tables
  cpds = []
  for node_alias, rp in raw_probs.items():
    variable = node_names[node_alias]
    variable_card = len(node_states[node_alias])
    values = rp["probs"].T
    state_names = {variable : node_states[node_alias]}
    if rp["parent_aliases"] is not None:
      evidence = [node_names[parent_alias] for parent_alias in rp["parent_aliases"]]
      evidence_card = [len(node_states[parent_alias]) for parent_alias in rp["parent_aliases"]]
      state_names.update({node_names[parent_alias] : node_states[parent_alias] for parent_alias in rp["parent_aliases"]})
    else:
      evidence = None
      evidence_card = None

    cpd = TabularCPD(variable, 
                     variable_card, 
                     values, 
                     evidence, 
                     evidence_card,
                     state_names)
    cpds.append(cpd)
  
  # Bake model
  edges = [(node_names[na1], node_names[na2]) for (na1, na2) in raw_edges]
  model = BayesianModel(edges)
  model.add_cpds(*cpds)
  model.check_model()

  return model
