import os
import time
import requests
import pandas as pd

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# Define AI model API endpoints (Updated Gemini endpoint)
API_ENDPOINTS = {
    "OpenAI GPT-4o Mini": "https://api.openai.com/v1/chat/completions",
    "DeepSeek V3": "https://api.deepseek.com/v1/chat/completions",
    "Llama 3.3 70B Instruct": "https://api.llama-api.com/chat/completions",
    "Gemini 2.0 Flash": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyAVYTHaz3qfWB72Qf2ZSj9lbIc6yeZ9nZA"
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


# Test questions that are based off of material found within the Rosen Discrete Mathmatics textbook
testQuestions = [
    "What are the main differences between a subset and a proper subset?",
    "How do I find the power set of a given set?",
    "Can you explain the principle of inclusion-exclusion with an example?",
    "What is the difference between a tautology, a contradiction, and a contingency",
    "How do I calculate the number of ways to distribute n identical objects into k distinct bins?",
    "How do I determine if a given graph is Eulerian or Hamiltonian?",
    "How do I prove that the sum of any two even numbers is always even?",
    "How do I solve the recurrence relation T(n) = 2T(n/2) + n using the Master Theorem?",
    "How do I construct a deterministic finite automaton (DFA) for a given regular expression",
    "What is the pumping lemma, and how is it used to prove that a language is not regular?",
    "Can you explain the difference between a context-free language and a regular language?"
]




# Initialize and handle queries for each model

def handleGPTPayload(question):

    # Initialize GPT payload
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 5000
    }

    return payload



def handleLlamaPayload(question):

    # Initialize LLama payload
    payload = {
        "model": "llama3.3-70b",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 5000
    }

    return payload




def handleDeepseekPayload(question):

    # Initialize Deepseek payload
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 5000
    }

    return payload




def handleGeminiPayload(question):

    # Initialize Gemini Payload
    payload = {
        "contents": [{"parts": [{"text": question}]}]
    }

    return payload





# Function to handle a question depending on the model passed through
def handleUserQuery(modelName, question):


    # Handle RAG preprocessing
    relevantDocuments = ragPreprocessing(question)
    print(f"The size of the relevant docs is: {len(relevantDocuments)}")

    modelQuestion = "Here are some documents that might help answer the question: " + question + "\n\n\n Relevant documents:\n" + "\n\n".join([doc.page_content for doc in relevantDocuments]) + "\n\nPlease provide an answer based on only the provided docs. If the answer is not found in the documents respond with im not sure if and only if you are unable to properly use your general knowledge base to solve the question. The relevant documents retrieved can be related to whatever is found, use it as context when asnwering a query to tailer an answer to the documents youve been fed. It does not need to be a word for word match"


    # Determine the model being used to generate appropriate payload
    if modelName == "OpenAI GPT-4o Mini":
        payload = handleGPTPayload(modelQuestion)

    elif modelName == "DeepSeek V3":
        payload = handleDeepseekPayload(modelQuestion)

    elif modelName == "Llama 3.3 70B Instruct":
        payload = handleLlamaPayload(modelQuestion)

    elif modelName == "Gemini 2.0 Flash":
        payload = handleGeminiPayload(modelQuestion)

    else:
        return float('inf'), "Invalid Model"  # Handle incorrect model names



    time.sleep(1.5)  # Wait 1.5 seconds before sending the next request


    # After we structure payload -> Go about handling request
    print("Testing model: " + modelName)
    startTime = time.time()

    # Trying to submit a post request of API provider
    try:
        response = requests.post(API_ENDPOINTS[modelName], headers=API_HEADERS[modelName], json=payload)
        endTime = time.time()
        latency = (endTime - startTime) * 1000

        # Debugging: Print API response
        print(f"\nResponse from {modelName}:")
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")

        if response.status_code == 200:
            response_json = response.json()
            if modelName == "Gemini 2.0 Flash": # Gemini response object is a little different than the other models
                answer = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No Response")
            else:
                answer = response_json.get("choices", [{}])[0].get("message", {}).get("content", "No Response")
            return round(latency, 2), answer
        else:
            return round(latency, 2), f"Error: {response.status_code}"
    except Exception as e:
        return float('inf'), f"Error: {e}"





# Handling the storing of external documents and vector store creation
# Handling the storing of external documents and vector store creation
def ragPreprocessing(question):
    # Define directory containing text file and the persistent directory
    currentDir = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(currentDir, "Books", "RosenDiscreteMath.txt")
    persistentDirectory = os.path.join(currentDir, "db", "chroma_db")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key="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"  # Use environment variable for security
    )

    # Check if the vector store exists
    if os.path.exists(persistentDirectory):
        print(" Vector store exists. Loading database...")
        database = Chroma(persist_directory=persistentDirectory, embedding_function=embeddings)

        # Check if the database is actually populated
        num_documents = database._collection.count()
        print(f" Found {num_documents} documents in ChromaDB.")

        if num_documents == 0:
            print(" ChromaDB is empty. Rebuilding vector store...")
            rebuild_vector_store(filePath, persistentDirectory, embeddings)
        else:
            print(" ChromaDB is populated. Proceeding with retrieval.")

    else:
        print("🚀 Persistent directory does not exist. Creating vector store...")
        rebuild_vector_store(filePath, persistentDirectory, embeddings)

    # Create a retriever
    retriever = database.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3}
    )

    return retriever.invoke(question)  # Returning retrieved documents


def rebuild_vector_store(filePath, persistentDirectory, embeddings):
    """Rebuilds the Chroma vector store from scratch if it's empty or missing."""
    if not os.path.exists(filePath):
        raise FileNotFoundError(f" The file {filePath} does not exist. Please check the path.")

    # Load and split documents
    loader = TextLoader(filePath)
    documents = loader.load()
    textSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = textSplitter.split_documents(documents)

    print("\n Creating new embeddings...")
    database = Chroma.from_documents(docs, embeddings, persist_directory=persistentDirectory)
    print(f" Successfully stored {len(docs)} documents in ChromaDB.")



# Actual main logic for testing each model through a series of questions

outputResults = []

# Iterate through each test question

for question in testQuestions:

    modelOutput = {"Question": question}  # Creating a dictionary for each test question

    # Iterate through the keys (model name) of API endpoints
    for model in API_ENDPOINTS.keys():

        # For each question, we need to run it through all 4 models
        latency, response = handleUserQuery(model, question)
        modelOutput[f"{model} (Latency)"] = f"{latency} ms"
        modelOutput[f"{model} (Response)"] = response

    outputResults.append(modelOutput)  # Append results of question for every model to a list



# Creating a dataframe object
dataframeResults = pd.DataFrame(outputResults)

# Converting dataframe object into an excel and csv file for comparasion
dataframeResults.to_csv("model_responsesRAG.csv", index=False)
dataframeResults.to_excel("model_responsesRAG.xlsx", index=False)









