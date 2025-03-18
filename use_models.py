import joblib
import numpy as np

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

# Function to load the appropriate model and scaler based on age
def get_model_and_scaler_for_age(age):
    if age <= 10:
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

# Function to predict ASD traits based on user input
def predict_asd(age):
    model, scaler, age_group = get_model_and_scaler_for_age(age)
    
    # Ask the relevant questions based on age group
    print(f"\nPlease answer the following questions for the {age_group.replace('_', ' ')} age group with 'yes' or 'no':\n")
    user_responses = []
    for question in questions[age_group]:
        while True:
            response = input(question + " (yes/no): ").strip().lower()
            if response in ['yes', 'no']:
                user_responses.append(1 if response == 'yes' else 0)
                break
            else:
                print("Invalid response. Please answer with 'yes' or 'no'.")
    
    # Transform user responses
    user_responses = np.array(user_responses).reshape(1, -1)
    user_responses = scaler.transform(user_responses)
    
    # Predict using the model
    prediction = model.predict(user_responses)
    
    if prediction[0] == 1:
        print("\nBased on your responses,  you might have ASD traits.")
    else:
        print("\nBased on your responses, it is less likely that you have ASD traits.")

# Main function to run the program
if __name__ == "__main__":
    age = int(input("Enter your age: "))
    predict_asd(age)
