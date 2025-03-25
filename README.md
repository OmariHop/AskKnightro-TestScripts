
# AskKnightro Test Scripts

This is a Github Repository for testing the different models that best fit the use case for AskKnightro-an Ai powered TA assistant

## Installation

For this project, we are going to need to install lang chain and associated depdencies

```bash
pip install pandas requests langchain-text-splitters langchain-community  langchain-openai chromadb openpyxl
```
    
## Handling API request for each model

For OpenAI, LLama, and Deepseek, they follow the same API payload and endpoint pattern. Insert your API Bearer tokens into the Authorization section of payload where it states *insert here*

```http
  API_ENDPOINTS = {
    "OpenAI GPT-4o Mini": "https://api.openai.com/v1/chat/completions",
    "DeepSeek V3": "https://api.deepseek.com/v1/chat/completions",
    "Llama 3.3 70B Instruct": "https://api.llama-api.com/chat/completions"
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
```


Google Gemini is handled differently where the API token is in the url of the request

```http
"Gemini 2.0 Flash": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=*insert key*
```

After API keys are provided, running testWithRAG.py and test.py should give you the outputs of each model in their respective excell spreadsheet

