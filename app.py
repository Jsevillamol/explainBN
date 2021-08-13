from flask import Flask
from flask import request
from flask import render_template
from flask_ngrok import run_with_ngrok

import numpy as np

from explainBN.load_network import load_network
from explainBN.scoring_table import generate_scoring_table
from explainBN.interpretation import draw_model, read_scoring_table
from explainBN.utilities import random_evidence, explain_evidence, prob, from_logodd_to_prob, introduce_error

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def my_form():
    
    # Load model
    model, target, evidence_nodes = load_network("asia")
    bn_graph_fn = "graph.png"
    draw_model(model, output_fn = f"static/{bn_graph_fn}")
    variable_description = model.variable_description.to_html(classes='data')
    scoring_table = generate_scoring_table(model, target, evidence_nodes)
    
    # Treatment no mistake
    evidence = random_evidence(model, evidence_nodes)
    
    evidence_explanation = explain_evidence(model, evidence)
    html_evidence_treatment_no_mistake = \
        render_template("evidence.html",
                        evidence_explanation = evidence_explanation,
                        )
    
    interactive_output = read_scoring_table(model, target, evidence, 
                                            scoring_table, 
                                            interactive = True,
                                            interactive_output_prefix = 'io_no_mistake')
    html_explanation_treatment_no_mistake = \
        render_template("treatment.html", 
                        html_element_id = 'explanation_treatment_no_mistake',
                        interactive_output = interactive_output,
                        )
    
    p_target = from_logodd_to_prob(read_scoring_table(model, target, evidence, scoring_table))
    html_survey_treatment_no_mistake = \
        render_template("bn_survey.html",
                        html_element_id = 'survey_treatment_no_mistake',
                        network_name = 'asia',
                        p_target = p_target,
                        is_treatment = 'true',
                        mistake_location = "no mistake",
                        )
    
    # Control no mistake
    evidence = random_evidence(model, evidence_nodes)
    
    evidence_explanation = explain_evidence(model, evidence)
    html_evidence_control_no_mistake = \
        render_template("evidence.html",
                        evidence_explanation = evidence_explanation,
                        )
    
    html_explanation_control_no_mistake = \
        render_template("control.html",
                        html_element_id = 'explanation_control_no_mistake',
                        bn_model = model,
                        squeeze_fn = np.squeeze, # Yes I know this is very hacky
                        )
    
    p_target = prob(model, target, evidence)
    html_survey_control_no_mistake = \
        render_template("bn_survey.html",
                        html_element_id = 'survey_control_no_mistake',
                        network_name = 'asia',
                        p_target = p_target,
                        is_treatment = 'false',
                        mistake_location = "no mistake",
                        )
    
    # Treatment mistake
    
    perturbed_model, perturbed_node = introduce_error(model, target, evidence)
    scoring_table = generate_scoring_table(model, target, evidence_nodes)
    evidence = random_evidence(model, evidence_nodes)
    
    evidence_explanation = explain_evidence(perturbed_model, evidence)
    html_evidence_treatment_mistake = \
        render_template("evidence.html",
                        evidence_explanation = evidence_explanation,
                        )
    
    interactive_output = read_scoring_table(perturbed_model, target, evidence, 
                                            scoring_table, 
                                            interactive = True,
                                            interactive_output_prefix = 'io_mistake')
    html_explanation_treatment_mistake = \
        render_template("treatment.html", 
                        html_element_id = 'explanation_treatment_mistake',
                        interactive_output = interactive_output,
                        )
    
    p_target = from_logodd_to_prob(read_scoring_table(perturbed_model, target, evidence, scoring_table))
    html_survey_treatment_mistake = \
        render_template("bn_survey.html",
                        html_element_id = 'survey_treatment_mistake',
                        network_name = 'asia',
                        p_target = p_target,
                        is_treatment = 'true',
                        mistake_location = str(perturbed_node),
                        )
    
    
    # Control mistake
    
    perturbed_model, perturbed_node = introduce_error(model, target, evidence)
    scoring_table = generate_scoring_table(model, target, evidence_nodes)
    evidence = random_evidence(model, evidence_nodes)
    
    evidence_explanation = explain_evidence(model, evidence)
    html_evidence_control_mistake = \
        render_template("evidence.html",
                        evidence_explanation = evidence_explanation,
                        )
    
    html_explanation_control_mistake = \
        render_template("control.html",
                        html_element_id = 'explanation_control_mistake',
                        bn_model = model,
                        squeeze_fn = np.squeeze, # Yes I know this is very hacky
                        )
    
    p_target = prob(perturbed_model, target, evidence)
    html_survey_control_mistake = \
        render_template("bn_survey.html",
                        html_element_id = 'survey_control_mistake',
                        network_name = 'asia',
                        p_target = p_target,
                        is_treatment = 'false',
                        mistake_location = str(perturbed_node),
                        )
    
    
    # Construct and return HTML template
    
    return render_template("template.html",
                           bn_graph=bn_graph_fn,
                           variable_description=variable_description,
                           
                           # Treatment no mistake components
                           html_evidence_treatment_no_mistake = html_evidence_treatment_no_mistake,
                           html_explanation_treatment_no_mistake = html_explanation_treatment_no_mistake,
                           html_survey_treatment_no_mistake = html_survey_treatment_no_mistake,
                           
                           # Control no mistake components
                           html_evidence_control_no_mistake = html_evidence_control_no_mistake,
                           html_explanation_control_no_mistake = html_explanation_control_no_mistake,
                           html_survey_control_no_mistake = html_survey_control_no_mistake,
                           
                           # Treatment mistake components
                           html_evidence_treatment_mistake = html_evidence_treatment_mistake,
                           html_explanation_treatment_mistake = html_explanation_treatment_mistake,
                           html_survey_treatment_mistake = html_survey_treatment_mistake,
                           
                           # Control mistake components
                           html_evidence_control_mistake = html_evidence_control_mistake,
                           html_explanation_control_mistake = html_explanation_control_mistake,
                           html_survey_control_mistake = html_survey_control_mistake,
                           
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