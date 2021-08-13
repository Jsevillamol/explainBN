import zipfile
import os
import pandas as pd

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
    fn = f"exampleBNs/{network_name}.bif"
    reader = BIFReader(fn)
    
  model = reader.get_model()
  model.states = reader.get_states()

  # Dictionary of targets and evidence_nodes
  NETWORK_NODES = {
      "asia" : {
          "target" : ("lung", "yes"),
          "evidence_nodes" : {'asia', 'tub', 'smoke', 'bronc', 
                              'xray', 'dysp'}
      },
      "cancer" : {
          "target" : ("Cancer", "True"),
          #"evidence_nodes" : {'Pollution', 'Smoker', 'Xray', 'Dyspnoea', 'Cancer'}
      },
      "earthquake" : {
          "target" : ("Earthquake", "True"),
          "evidence_nodes" : {'Burglary', 'Alarm', 'JohnCalls', 'MaryCalls'}
      },
      "child" : {
          "target" : ("Sick", "yes"),
      },
      "three_nations" : {
          "target" : ("Trubia launched missile", "True")
      }
  }


  # Select observable nodes and target
  try:
    target = NETWORK_NODES[network_name]["target"]
  except KeyError:
    target_node = np.random.choice(model.nodes())
    target = (target_node, np.random.choice(model.states[target_node]))

  try:
    evidence_nodes = NETWORK_NODES[network_name]["evidence_nodes"]
  except KeyError:
    evidence_nodes = list(model.nodes())
    evidence_nodes.remove(target[0])

  # Check validity of target and evidence nodes
  assert target[0] in model.nodes() and target[1] in model.states[target[0]]
  for evidence_node in evidence_nodes:
    assert evidence_node in model.nodes()

  if verbose:
    print(f"Number of nodes = {len(model.nodes)}")
    print(f"Number of edges = {len(model.edges)}")

    print(f"target = {target}")
    print(f"evidence_nodes = {evidence_nodes}")

  # Add explanation of nodes
  if network_name == 'asia':

    model.variable_description = pd.DataFrame([
        ("smoke", "Whether the patient smokes"),
        ("asia", "Whether the patient has recently been to Asia"),
        ("lung", "Whether the patient has lung cancer"),
        ("bronc", "Whether the patient has bronchitis"),
        ("tub", "Whether the patient has tuberculosis"),
        ("lung_disease", "True if the patient has either lung cancer or tuberculosis"),
        ("xray", "Whether the patient's xray results show an abnormality"),
        ("dysp", "Whether the patient experiences shortness of breath (dyspnea)"),
    ], columns=['Variable', 'Meaning'])

    model.explanation_dictionary = {
        ('smoke', 'yes')  : {
            'explanation' : 'the patient smokes',
            'polarity'    : 'positive',
        },
        ('smoke', 'no')   : {
            'explanation' : 'the patient does not smoke',
            'contrastive_explanation' : 'the patient smokes',
            'polarity'    : 'negative',
        },
        ('asia', 'yes')  : {
            'explanation' : 'the patient has recently visited Asia',
            'polarity'    : 'positive',
        },
        ('asia', 'no')   : {
            'explanation' : 'the patient has not recently visited Asia',
            'contrastive_explanation' : 'the patient has recently visited Asia',
            'polarity'    : 'negative',
        },
        ('lung', 'yes')  : {
            'explanation' : 'the patient has lung cancer',
            'polarity'    : 'positive',
        },
        ('lung', 'no')   : {
            'explanation' : 'the patient does not have lung cancer',
            'contrastive_explanation' : 'the patient has lung cancer',
            'polarity'    : 'negative',
        },
        ('bronc', 'yes')  : {
            'explanation' : 'the patient has bronchitis',
            'polarity'    : 'positive',
        },
        ('bronc', 'no')   : {
            'explanation' : 'the patient does not have bronchitis',
            'contrastive_explanation' : 'the patient has bronchitis',
            'polarity'    : 'negative',
        },
        ('tub', 'yes')  : {
            'explanation' : 'the patient has tuberculosis',
            'polarity'    : 'positive',
        },
        ('tub', 'no')   : {
            'explanation' : 'the patient does not have tuberculosis',
            'contrastive_explanation' : 'the patient has tuberculosis',
            'polarity'    : 'negative',
        },
        ('lung_disease', 'yes')   : {
            'explanation' : 'the patient has a serious lung disease',
            'polarity'    : 'positive',
        },
        ('lung_disease', 'no')   : {
            'explanation' : 'the patient does not have a serious lung disease',
            'polarity'    : 'positive',
        },
        ('xray', 'abnormal')   : {
            'explanation' : 'the patient\'s xray results show an abnormality',
            'polarity'    : 'positive',
        },
        ('xray', 'normal')   : {
            'explanation' : 'the patient\'s xray results are normal',
            'polarity'    : 'positive',
        },
        ('dysp', 'yes')  : {
            'explanation' : 'the patient experiences shortness of breath',
            'polarity'    : 'positive',
        },
        ('dysp', 'no')   : {
            'explanation' : 'the patient does not experience shortness of breath',
            'contrastive_explanation' : 'the patient experiences shortness of breath',
            'polarity'    : 'negative',
        },
    }

  elif network_name == 'earthquake':
    model.explanation_dictionary = {
        ('Burglary', 'True')    : "a burglary happened" ,
        ('Burglary', 'False')   : "no burglary occured",
        ('Earthquake', 'True')  : "an earthquake has happened",
        ('Earthquake', 'False') : "there was no earthquake",
        ('Alarm', 'True')       : "the alarm rang",
        ('Alarm', 'False')      : "the alarm didn't ring",
        ('JohnCalls', 'True')   : "John called",
        ('JohnCalls', 'False')  : "John didn't call",
        ('MaryCalls', 'True')   : "Mary called",
        ('MaryCalls', 'False')  : "Mary didn't call",
    }

  elif network_name == "three_nations":
    model.explanation_dictionary = {
   ('Oclar Expert 1 Report','Neg'): {
        'explanation': 'Oclar expert 1 confirms they had missiles', 
        'polarity': 'positive'},
   ('Oclar Expert 1 Report','Pos'): {
       'explanation': 'Oclar expert 1 confirms they had missiles', 
       'polarity': 'positive'},
   ('Oclar Expert 2 Report','Neg'): {
       'explanation': 'Oclar expert 2 denies they had missiles', 
       'polarity': 'positive'},
   ('Oclar Expert 2 Report','Pos'): {
       'explanation': 'Oclar expert 2 confirms they had missiles', 
       'polarity': 'positive'},
   ('Oclar has weapons','False'): {
       'contrastive_explanation': 'Oclar has missiles', 
       'explanation': 'Oclar does not have missiles', 
       'polarity': 'negative'},
   ('Oclar has weapons','True'): {
       'explanation': 'Oclar has missiles', 
       'polarity': 'positive'},
   ('Oclar launched missile','False'): {
       'contrastive_explanation': 'Oclar did not launch a missile', 
       'explanation': 'Oclar launched a missile', 
       'polarity': 'negative'},
   ('Oclar launched missile','True'): {
       'explanation': 'Oclar launched a missile', 
       'polarity': 'positive'},
   ('Residue Detected','Neg'): {
       'contrastive_explanation': 'some missile residue was detected', 
       'explanation': 'no missile residue was detected', 
       'polarity': 'positive'},
   ('Residue Detected','Pos'): {
       'explanation': 'some missile residue was detected', 
       'polarity': 'positive'},
   ('Trubia Expert 1 Report','Neg'): {
       'explanation': 'expert 1 from Trubia denies they had weapons', 
       'polarity': 'positive'},
   ('Trubia Expert 1 Report','Pos'): {
       'explanation': 'expert 1 from Trubia confirms they have missiles', 
       'polarity': 'positive'},
   ('Trubia Expert 2 Report','Neg'): {
       'explanation': 'expert 2 from Trubia denies they had weapons', 
       'polarity': 'positive'},
   ('Trubia Expert 2 Report','Pos'): {
       'explanation': 'expert 2 from Trubia confirms they had weapons', 
       'polarity': 'positive'},
   ('Trubia has weapons','False'): {
       'contrastive_explanation': 'Trubia has missiles', 
       'explanation': 'Trubia does not have missiles', 
       'polarity': 'negative'},
   ('Trubia has weapons','True'): { 
       'explanation': 'Trubia has missiles', 
       'polarity': 'positive'},
   ('Trubia launched missile','False'): {
       'contrastive_explanation': 'Trubia launched a missile', 
       'explanation': 'Trubia did not launch a missile', 
       'polarity': 'negative'},
   ('Trubia launched missile','True'): {
       'explanation': 'Trubia launched a missile', 
       'polarity': 'positive'}
     }
    model.variable_description = pd.DataFrame([
      ("Oclar has weapons", "Whether Oclar has missiles."),
      ("Oclar launched missile", "Whether Oclar launched a missile"),
      ("Oclar Expert 2 Report", "Whether expert 2 from Oclar denies having missiles."),
      ("Oclar Expert 1 Report", "Whether expert 1 from Oclar denies having missiles."),
      ("Trubia has weapons","Whether Trubia has missiles" ),
      ("Trubia launched missile", "Whether Trubia launched a missile"),
      ("Residue Detected", "Whether missile residue was detected"),
      ("Trubia Expert 1 Report", "Whether expert 1 from Trubia denies having missiles"),
      ("Trubia Expert 2 Report", "Whether expert 2 from Trubia denies having missiles"),
    ], columns=['Variable', 'Meaning'])

  else:
    model.variable_description = \
      pd.DataFrame([(node, node) for node in model.nodes])
    model.explanation_dictionary = {
        (node, state) : {
            "explanation" : f"{node} = {state}",
            "contrastive_explanation" : f"not {node} = {state}",
            "polarity" : "positive",
        }
        for node in model.nodes
        for state in model.states[node]
    }

  # Precompute baseline marginal distribution of target
  model.baselines = {}
  # model.baselines[target[0]] = VariableElimination(model).query(variables=[target[0]],
  #                                                               evidence={},
  #                                                               show_progress=False)

  v = VariableElimination(model)
  for node in model.nodes:
    model.baselines[node] = v.query(variables=[node], 
                                    evidence={}, 
                                    show_progress=False)
    
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
