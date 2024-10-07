import requests
import json

def query_model(prompt):
    url = 'http://localhost:5000/generate_sql'
    headers = {'Content-Type': 'application/json'}
    data = {'prompt': prompt}
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def main():
    print("Enter your SQL generation prompts. Type 'exit' to quit.")
    while True:
        prompt = input("Enter prompt: ")
        if prompt.lower() == 'exit':
            break
        response = query_model(prompt)
        print(json.dumps(response, indent=4))

if __name__ == "__main__":
    main()
