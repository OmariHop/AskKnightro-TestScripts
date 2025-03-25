import time
import requests
import random
import pandas as pd


from concurrent.futures import ThreadPoolExecutor
from google import genai


# Define AI model API endpoints (Updated Gemini endpoint)
API_ENDPOINTS = {
    "OpenAI GPT-4o Mini": "https://api.openai.com/v1/chat/completions",
    "DeepSeek V3": "https://api.deepseek.com/v1/chat/completions",
    "Llama 3.3 70B Instruct": "https://api.llama-api.com/chat/completions",
    "Gemini 2.0 Flash": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=*insert here*"
}

# Define API headers (Ensure proper 'Bearer' format)
API_HEADERS = {
    "OpenAI GPT-4o Mini": {
        "Authorization": "Bearer *insert here*",
        "Content-Type": "application/json"
    },
    "DeepSeek V3": {
        "Authorization": "Bearer *insert here*",
        "Content-Type": "application/json"
    },
    "Llama 3.3 70B Instruct": {
        "Authorization": "Bearer *insert here*",
        "Content-Type": "application/json"
    },
    "Gemini 2.0 Flash": {
        "Content-Type": "application/json"
    }
}

# Standardized test questions
TEST_QUESTIONS = [
    "What is the Pythagorean theorem?",
    "Summarize the causes of World War I in two sentences.",
    "Explain Newton’s third law of motion.",
    "Translate 'Hello, how are you?' to Spanish.",
    "Solve the equation: 2x + 3 = 11"
]

# Function to test model latency and response
def test_model(model_name, question):
    if model_name == "OpenAI GPT-4o Mini":
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 100
        }
    elif model_name == "DeepSeek V3":
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 200
        }
    elif model_name == "Llama 3.3 70B Instruct":
        payload = {
            "model": "llama3.3-70b",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 200
        }
    elif model_name == "Gemini 2.0 Flash":
        payload = {
            "contents": [{"parts": [{"text": question}]}]
        }
    else:
        return float('inf'), "Invalid Model"  # Handle incorrect model names

    start_time = time.time()
    print("Testing model: " + model_name)

    try:
        response = requests.post(API_ENDPOINTS[model_name], headers=API_HEADERS[model_name], json=payload)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        # Debugging: Print API response
        print(f"\nResponse from {model_name}:")
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")

        if response.status_code == 200:
            response_json = response.json()
            if model_name == "Gemini 2.0 Flash": # Gemini response object is a little different than the other models
                answer = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No Response")
            else:
                answer = response_json.get("choices", [{}])[0].get("message", {}).get("content", "No Response")
            return round(latency, 2), answer
        else:
            return round(latency, 2), f"Error: {response.status_code}"
    except Exception as e:
        return float('inf'), f"Error: {e}"

# Collect results
output_results = []

for question in TEST_QUESTIONS:
    model_outputs = {"Question": question}

    for model in API_ENDPOINTS.keys():
        latency, response = test_model(model, question)
        model_outputs[f"{model} (Latency)"] = f"{latency} ms"
        model_outputs[f"{model} (Response)"] = response

    output_results.append(model_outputs)

# Convert results to DataFrame
df_results = pd.DataFrame(output_results)

# Save results to CSV and Excel
df_results.to_csv("model_responses.csv", index=False)
df_results.to_excel("model_responses.xlsx", index=False)