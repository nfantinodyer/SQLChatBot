from flask import Flask, request, jsonify
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the trained model and tokenizer
model_path = './fine_tuned_t5_sql'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

app = Flask(__name__)

@app.route('/generate_sql', methods=['POST'])
def generate_sql():
    if request.is_json:
        try:
            content = request.get_json()
            prompt = content['prompt']
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                inputs,
                max_length=80,  # Adjust as needed
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return jsonify({'sql_query': sql_query})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({"error": "Request must be JSON"}), 415

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
