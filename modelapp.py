# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # Define the questions for each age group
# questions = {
#     "children": [
#         "Does your child avoid eye contact?",
#         "Does your child show interest in other children?",
#         "Does your child engage in pretend play?",
#         "Does your child repeat the same actions or words?",
#         "Does your child have trouble understanding others' feelings?",
#         "Does your child have trouble adjusting to changes in routine?",
#         "Does your child flap their hands or spin objects?",
#         "Does your child have difficulty with social interactions?",
#         "Does your child appear oversensitive to noises or lights?",
#         "Does your child have specific interests or routines?"
#     ],
#     "adolescents": [
#         "Do you have difficulty maintaining friendships?",
#         "Do you prefer to spend time alone rather than with friends?",
#         "Do you find it hard to understand what others are thinking or feeling?",
#         "Do you have trouble understanding jokes or sarcasm?",
#         "Do you find it difficult to adapt to changes?",
#         "Do you engage in repetitive behaviors or have specific routines?",
#         "Do you have intense interests in specific topics?",
#         "Do you find social situations overwhelming?",
#         "Do you avoid eye contact?",
#         "Do you have difficulty with small talk or casual conversations?"
#     ],
#     "young_adults": [
#         "Do you find it difficult to understand other people's emotions?",
#         "Do you prefer to follow routines and find change difficult?",
#         "Do you avoid social situations or find them overwhelming?",
#         "Do you struggle with making eye contact during conversations?",
#         "Do you have specific hobbies or interests that you focus on intensely?",
#         "Do you find it challenging to engage in small talk?",
#         "Do you often miss social cues, such as when someone is being sarcastic?",
#         "Do you feel uncomfortable in group settings?",
#         "Do you prefer to be alone rather than with others?",
#         "Do you find loud noises or bright lights distressing?"
#     ],
#     "adults": [
#         "Do you often prefer to be alone rather than with others?",
#         "Do you find it challenging to understand social cues?",
#         "Do you have specific routines or rituals that you prefer not to break?",
#         "Do you avoid eye contact in conversations?",
#         "Do you feel overwhelmed in social situations?",
#         "Do you have intense interests or hobbies?",
#         "Do you struggle with changes to your routine?",
#         "Do you find it hard to make friends?",
#         "Do you have difficulty understanding jokes or sarcasm?",
#         "Do you feel uncomfortable in new situations or with unfamiliar people?"
#     ]
# }

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html', 
#                            children_questions=questions['children'], 
#                            adolescent_questions=questions['adolescents'],
#                            young_adults_questions=questions['young_adults'],
#                            adult_questions=questions['adults'])

# def get_model_and_scaler_for_age(age):
#     if age <= 10:
#         model_path = 'models/children_asd_model.pkl'
#         age_group = 'children'
#     elif 11 <= age <= 17:
#         model_path = 'models/adolescent_asd_model.pkl'
#         age_group = 'adolescents'
#     elif 18 <= age <= 35:
#         model_path = 'models/young_asd_model.pkl'
#         age_group = 'young_adults'
#     else:
#         model_path = 'models/adult_asd_model.pkl'
#         age_group = 'adults'
    
#     model = joblib.load(model_path)
#     scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
    
#     return model, scaler, age_group

# @app.route('/predict', methods=['POST'])
# def predict():
#     age = int(request.form['age'])
#     model, scaler, age_group = get_model_and_scaler_for_age(age)
    
#     user_responses = []
#     for i in range(1, 11):
#         response = request.form.get(f'Q{i}')
#         user_responses.append(int(response))
    
#     # Create a DataFrame with the same column names used during training
#     feature_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10_Autism_Spectrum_Quotient']
#     user_responses_df = pd.DataFrame([user_responses], columns=feature_names)
    
#     # Scale the user responses
#     user_responses_scaled = scaler.transform(user_responses_df)
    
#     # Predict using the model
#     prediction = model.predict(user_responses_scaled)
    
#     if prediction[0] == 1:
#         result_text = "Based on your responses, there is a likelihood that you might have ASD traits. Please consider seeking professional advice."
#     else:
#         result_text = "Based on your responses, it is less likely that you have ASD traits."
    
#     return render_template('result.html', result=result_text)


# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Define the questions for each age group
questions = {
    "children": [
        "Does your child avoid eye contact?",
        "Does your child show interest in other children?",
        "Does your child engage in pretend play?",
        "Does your child repeat the same actions or words?",
        "Does your child have trouble understanding others' feelings?",
        "Does your child have trouble adjusting to changes in routine?",
        "Does your child flap their hands or spin objects?",
        "Does your child have difficulty with social interactions?",
        "Does your child appear oversensitive to noises or lights?",
        "Does your child have specific interests or routines?"
    ],
    "adolescents": [
        "Do you have difficulty maintaining friendships?",
        "Do you prefer to spend time alone rather than with friends?",
        "Do you find it hard to understand what others are thinking or feeling?",
        "Do you have trouble understanding jokes or sarcasm?",
        "Do you find it difficult to adapt to changes?",
        "Do you engage in repetitive behaviors or have specific routines?",
        "Do you have intense interests in specific topics?",
        "Do you find social situations overwhelming?",
        "Do you avoid eye contact?",
        "Do you have difficulty with small talk or casual conversations?"
    ],
    "young_adults": [
        "Do you find it difficult to understand other people's emotions?",
        "Do you prefer to follow routines and find change difficult?",
        "Do you avoid social situations or find them overwhelming?",
        "Do you struggle with making eye contact during conversations?",
        "Do you have specific hobbies or interests that you focus on intensely?",
        "Do you find it challenging to engage in small talk?",
        "Do you often miss social cues, such as when someone is being sarcastic?",
        "Do you feel uncomfortable in group settings?",
        "Do you prefer to be alone rather than with others?",
        "Do you find loud noises or bright lights distressing?"
    ],
    "adults": [
        "Do you often prefer to be alone rather than with others?",
        "Do you find it challenging to understand social cues?",
        "Do you have specific routines or rituals that you prefer not to break?",
        "Do you avoid eye contact in conversations?",
        "Do you feel overwhelmed in social situations?",
        "Do you have intense interests or hobbies?",
        "Do you struggle with changes to your routine?",
        "Do you find it hard to make friends?",
        "Do you have difficulty understanding jokes or sarcasm?",
        "Do you feel uncomfortable in new situations or with unfamiliar people?"
    ]
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', 
                           children_questions=questions['children'], 
                           adolescent_questions=questions['adolescents'],
                           young_adults_questions=questions['young_adults'],
                           adult_questions=questions['adults'])

def get_model_and_scaler_for_age(age):
    if 0< age <= 10:
        model_path = 'models/children_asd_model.pkl'
        age_group = 'children'
    elif 11 <= age <= 17:
        model_path = 'models/adolescent_asd_model.pkl'
        age_group = 'adolescents'
    elif 18 <= age <= 35:
        model_path = 'models/young_asd_model.pkl'
        age_group = 'young_adults'
    else:
        model_path = 'models/adult_asd_model.pkl'
        age_group = 'adults'
    
    model = joblib.load(model_path)
    scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
    
    return model, scaler, age_group

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
    except KeyError:
        return "Missing age parameter!", 400
    
    model, scaler, age_group = get_model_and_scaler_for_age(age)
    
    user_responses = []
    for i in range(1, 11):
        response = request.form.get(f'Q{i}')
        if response is None:
            return f"Missing answer for question {i}!", 400
        user_responses.append(int(response))
    
    # Create a DataFrame with the same column names used during training
    feature_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10_Autism_Spectrum_Quotient']
    user_responses_df = pd.DataFrame([user_responses], columns=feature_names)
    
    # Scale the user responses
    user_responses_scaled = scaler.transform(user_responses_df)
    
    # Predict using the model
    prediction = model.predict(user_responses_scaled)
    
    if prediction[0] == 1:
        result_text = "Your responses suggest possible ASD traits. "
    else:
        result_text = "Based on your responses, ASD traits are absent."
    
    return render_template('result.html', result=result_text)

if __name__ == "__main__":
    app.run(debug=True)
