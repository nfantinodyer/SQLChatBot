import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, request, jsonify

# Load the trained model and tokenizer
model_path = './fine_tuned_t5_sql'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def generate_sql_query(prompt):
    # Encode the prompt to tensor
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate outputs
    outputs = model.generate(inputs, max_length=256, num_beams=5, early_stopping=True)
    
    # Decode the generated ids to string
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

def main():
    while True:
        user_input = input("Enter your SQL query prompt or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        
        # Generate model's prediction
        generated_query = generate_sql_query(user_input)
        print("Model's SQL Prediction:", generated_query)
        
        # User provides the correct SQL if the prediction is wrong
        correct_sql = input("Enter the correct SQL if wrong, or just press enter if correct: ")
        
        # Save interaction
        if correct_sql:
            save_interaction(user_input, correct_sql)
        else:
            save_interaction(user_input, generated_query)

def save_interaction(prompt, sql):
    # This function saves the interactions to a JSON file for later training
    data = {'prompt': prompt, 'sql': sql}
    with open('trainingData.json', 'a') as f:
        json.dump(data, f)
        f.write('\n')

    print("Interaction saved.")

if __name__ == "__main__":
    main()
