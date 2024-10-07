from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load tokenizer and model from your fine-tuned directory
tokenizer = T5Tokenizer.from_pretrained('./fine_tuned_t5_sql')
model = T5ForConditionalGeneration.from_pretrained('./fine_tuned_t5_sql')

# Define your prompt
prompt = "Generate a SQL query to find all customers in New York"

# Encode the input and prepare it for the model
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Example debugging outputs during generation
outputs = model.generate(
    input_ids,
    max_length=80,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True,
    output_scores=True,  # To get scores of generated tokens
    return_dict_in_generate=True  # To get a more detailed output with attention and scores
)

# Printing additional information for debugging
print("Generated SQL:", tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
print("Generation Scores:", outputs.scores)
