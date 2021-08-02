from .utilities import create_trivial_argument, get_endpoints, prob, is_sub_argument, get_target

import networkx as nx
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# INTERPRETING SCORING TABLE

def read_scoring_table(model, target, evidence, scoring_table, interactive = False):
  # Find baseline entry and drop it from table
  baseline_entry = scoring_table.iloc[0]
  score = baseline_entry["score"]

  scoring_table = scoring_table.drop(0, axis=0)

  # Filter rows that apply to this input
  filter_fn = lambda row : \
    (set(row['observations'].items()) <= set(evidence.items())) and \
    (set(evidence.keys()).isdisjoint(set(row["argument"].nodes) - set(row["observations"].keys())))
    
  mask = scoring_table.apply(filter_fn, axis=1)
  scoring_table = scoring_table[mask]

  # Filter less specific rows in favor of more specific joint rows
  minimal_rows = []
  for index, row in scoring_table.iterrows():
    for argument in scoring_table["argument"]:
      if argument == row["argument"]: continue
      if is_sub_argument(row["argument"], argument): break
    else:
      minimal_rows.append(row)
  
  scoring_table = pd.DataFrame(minimal_rows)

  # Announce baseline in interactive mode
  if interactive:
    interactive_output = []
    arguments = [create_trivial_argument(target[0])]
    fn = f"graph_{len(interactive_output)}.png"
    draw_model(model, arguments, output_fn = f"static/{fn}")
    
    interactive_output.append({
      "text" : [baseline_entry["explanation"]],
      "img"  : fn,
      })
        

  # Process each relevant argument
  for index, row in scoring_table.iterrows():
    # Add precomputed score contribution to current score
    old_score = score
    score += row['score']
    
    if interactive:
      
      old_p = 1. / (1./np.exp(old_score) + 1.)
      p = 1. / (1./np.exp(score) + 1.)
      argument = row["explanation"]
      explanation = model.explanation_dictionary[target]["explanation"]
      commentary = f"This consideration changes the probability that {explanation} from {old_p*100:0.2f}% to {p*100:0.2f}%"
      text = row["explanation"] + [commentary]
      
      fn = f"graph_{len(interactive_output)}.png"
      draw_model(model, [row['argument']], output_fn = f"static/{fn}")
      
      arguments.append(row['argument'])
      
      interactive_output.append({
        "text" : text,
        "img"  : fn,
        })

  if interactive:
    
    p = 1 / (1 + 1 / np.exp(score))
    explanation = model.explanation_dictionary[target]["explanation"]
    
    text = f"After taking into account these {len(interactive_output)} arguments, we conclude that " + \
           f"the probability that {explanation} is {p*100:0.2f}%"
    
    fn = f"graph_{len(interactive_output)}.png"    
    draw_model(model, arguments, output_fn = f"static/{fn}")

    # p = prob(model, target, dict(evidence))
    # print(f"For comparison, the estimation of the probability using message passing is {p*100:0.2f}%")
    
    interactive_output.append({
        "text" : [text],
        "img"  : fn,
        })
      
    return interactive_output

  else: return score

# DRAWING GRAPHS WITH HIGHLIGHTED PATHS
swap = lambda edge : (edge[1], edge[0])

def draw_model(model, arguments=[], argument=None, output_fn = "static/graph.png"):
  G = model
  df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
  for row, data in nx.shortest_path_length(G):
    for col, dist in data.items():
        df.loc[row,col] = dist
  df = df.fillna(df.max().max())
  pos = nx.kamada_kawai_layout(G,dist=df.to_dict())

  if argument is not None:
    arguments = [argument]
  
  if len(arguments) > 0:
    nodes_to_color = []
    edges_to_color = []
    sources = []
    
    for argument in arguments:
      nodes_to_color += [node for node in argument if node in model]
      edges_and_swapped_edges = list(argument.to_undirected().edges) + \
                                [(n2,n1) for (n1,n2) in argument.to_undirected().edges]
      edges_to_color += [(node1, node2) 
                         for (node1, factor1) in edges_and_swapped_edges 
                         for (factor2, node2) in edges_and_swapped_edges
                         if isinstance(node1, str) and isinstance(node2, str) 
                         and node1 != node2 and factor1 == factor2]
      sources += get_endpoints(argument)
      
    target = get_target(argument)

    node_color_map = ['LightSalmon' if node == target else
                      'SkyBlue' if node in sources else 
                      'DodgerBlue' if node in nodes_to_color else 
                      'Gainsboro' 
                      for node in model]
        
    edge_color_map = ['CornflowerBlue' if edge in edges_to_color 
                              or swap(edge) in edges_to_color 
                      else 'DimGray' 
                      for edge in model.edges]
  else:
    node_color_map = 'white'
    edge_color_map = 'CornflowerBlue'
   
  f = plt.figure()

  nx.draw(G, pos, 
          node_size = 1600,
          width = 5,
          with_labels=True, 
          node_color = node_color_map,
          edge_color = edge_color_map,
          font_family = 'Arial',
          font_weight = 'bold')
          
  f.savefig(output_fn)
  
# CONTROL VISUALIZATION

def display_control_visualization(model, target, evidence_nodes):
  import ipywidgets
  evidence_widgets = {}
  for node in evidence_nodes:
    assert 'UNKNOWN' not in model.states[node]
    w = ipywidgets.Dropdown(
        options=model.states[node] + ['UNKNOWN'],
        value='UNKNOWN',
        description= node,
        disabled=False,
    )
    display(w)
    evidence_widgets[node] = w


  explanation = model.explanation_dictionary[target]["explanation"]
  p = prob(model, target, {})
  output_w = ipywidgets.Text(f"{p*100:0.2f}%", description=f"probability")
  output_w.disabled = True

  def on_button_bn_clicked(b):
    #clear_output()
    evidence = {node:w.value for node,w in evidence_widgets.items() 
                if w.value != 'UNKNOWN'}

    p = prob(model, target, evidence)

    output_w.disabled = False
    output_w.value = f"{p*100:0.2f}%"
    output_w.disabled = True

  button_bn = ipywidgets.Button(description='Enter evidence')
  button_bn.on_click(on_button_bn_clicked)

  display(button_bn)
  print(f"The probability that {explanation} is ")
  display(output_w)
