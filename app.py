from flask import Flask
from flask import request
from flask import render_template
from flask_ngrok import run_with_ngrok

import numpy as np

from explainBN.load_network import load_network
from explainBN.scoring_table import generate_scoring_table
from explainBN.interpretation import draw_model, read_scoring_table
from explainBN.utilities import random_evidence

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def my_form():
    model, target, evidence_nodes = load_network("asia")
    bn_graph_fn = "graph.png"
    draw_model(model, output_fn = f"static/{bn_graph_fn}")
    
    scoring_table = generate_scoring_table(model, 
                                           target, 
                                           evidence_nodes)
                                           
    evidence = random_evidence(model, evidence_nodes)
    
    interactive_output = read_scoring_table(model, target, evidence, 
                                            scoring_table, 
                                            interactive = True)
                                            
    variable_description = model.variable_description.to_html(classes='data')
    
    evidence_explanation = [model.explanation_dictionary[obs]["explanation"]
                            for obs in evidence.items()]
    
    return render_template("template.html",
                           bn_model = model,
                           bn_graph=bn_graph_fn,
                           variable_description=variable_description,
                           evidence_explanation = evidence_explanation,
                           interactive_output = interactive_output,
                           squeeze_fn = np.squeeze, # Yes I know this is very hacky
                           )

@app.route('/', methods=['POST'])
def my_form_post():
    text1 = request.form['text1']
    text2 = request.form['text2']
    if text1 == text2 :
        return "<h1>Plagiarism Detected !</h1>"
    else :
        return "<h1>No Plagiarism Detected !</h1>"

if __name__ == '__main__':
    app.run()