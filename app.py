# Import necessary libraries
import os
import time
import json
import streamlit as st

# Initialize OpenAI API key
import openai

# Set up OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Define assistant's name and instructions
name = "PharmaBot"
instructions = (
    "You are PharmaBot, an intelligent assistant designed to provide detailed information about pharmaceutical products. "
    "You assist users by answering questions about medications, their uses, side effects, interactions, dosage, and other relevant information. "
    "Ensure that all information provided is accurate, up-to-date, and sourced from reliable medical data. "
    "Always present information in a clear, professional, and empathetic manner. "
    "If you are unsure about an answer, recommend consulting a healthcare professional."
)

# Choose the model that supports function calling
model = "gpt-4o-mini"  # Update to a model that supports function calling

# Load and prepare pharmaceutical data for retrieval
@st.cache_data
def load_pharma_data():
    # Load the data
    with open('pharmaceutical_data.json', 'r', encoding='utf-8') as f:
        pharma_data = json.load(f)
    return pharma_data

pharma_data = load_pharma_data()

# Prepare the data for retrieval and create embeddings
@st.cache_resource
def prepare_embeddings(pharma_data):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import numpy as np

    # Combine all entries into a single text
    texts = []
    for entry in pharma_data:
        content = f"Name: {entry.get('name', '')}\n"
        content += f"Description: {entry.get('description', '')}\n"
        content += f"Uses: {entry.get('uses', '')}\n"
        content += f"Side Effects: {entry.get('side_effects', '')}\n"
        content += f"Dosage: {entry.get('dosage', '')}\n"
        content += f"Interactions: {entry.get('interactions', '')}\n"
        texts.append(content)

    # Split texts if necessary
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_texts = []
    for text in texts:
        split_texts.extend(text_splitter.split_text(text))

    # Function to get embeddings
    def get_embedding(text, model="text-embedding-ada-002"):
        response = openai.Embedding.create(input=[text], model=model)
        return response['data'][0]['embedding']

    # Create embeddings for each text chunk
    embeddings = [get_embedding(text) for text in split_texts]

    # Store the texts and their embeddings
    embeddings_data = list(zip(split_texts, embeddings))
    return embeddings_data

embeddings_data = prepare_embeddings(pharma_data)

# Function to perform retrieval
def retrieve_relevant_text(query):
    import numpy as np

    def get_embedding(text, model="text-embedding-ada-002"):
        response = openai.Embedding.create(input=[text], model=model)
        return response['data'][0]['embedding']

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    query_embedding = get_embedding(query)
    # Compute similarity with each text chunk
    similarities = [cosine_similarity(query_embedding, emb) for text, emb in embeddings_data]
    # Get the top 3 most similar texts
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
    most_similar_texts = [embeddings_data[i][0] for i in top_indices]
    return "\n\n".join(most_similar_texts)

# Define the function schema for retrieval
retrieval_function_schema = {
    "name": "retrieval",
    "description": "Retrieve relevant information from the pharmaceutical data.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's query to search for in the data."
            }
        },
        "required": ["query"]
    }
}

# Function to perform web search using Google Serper API (for the latest information)
def duckduckgo_search(query):
    import requests

    # Set up Serper API key from environment variable
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Serper API key not found. Please set the SERPER_API_KEY environment variable."

    url = 'https://google.serper.dev/search'
    payload = {
        'q': query
    }
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    response = requests.post(url, json=payload, headers=headers)
    results = response.json()

    # Extract snippets from search results
    snippets = []
    if 'organic' in results:
        for result in results['organic']:
            if 'snippet' in result:
                snippets.append(result['snippet'])

    return "\n".join(snippets)

# Define the function schema for DuckDuckGo search
ddg_function = {
    "name": "duckduckgo_search",
    "description": "Fetch up-to-date information by searching the web.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to use for fetching up-to-date information."
                }
            },
        "required": ["query"]
    }
}

# Function to interact with the assistant using all tools
def pharma_bot_assistant(query):
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": query}
    ]
    functions = [retrieval_function_schema, ddg_function]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call="auto"
    )
    message = response['choices'][0]['message']

    while True:
        if message.get("function_call"):
            # The assistant wants to call a function
            function_name = message["function_call"]["name"]
            function_args = json.loads(message["function_call"].get("arguments", "{}"))

            # Call the appropriate function
            if function_name == "retrieval":
                function_response = retrieve_relevant_text(function_args["query"])
            elif function_name == "duckduckgo_search":
                function_response = duckduckgo_search(function_args["query"])
            else:
                function_response = "Function not implemented."

            # Append the function call and response to the messages
            messages.append(message)
            messages.append({
                "role": "function",
                "name": function_name,
                "content": function_response
            })

            # Get the assistant's next response
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                functions=functions,
                function_call="auto"
            )
            message = response['choices'][0]['message']
        else:
            # The assistant has provided a final answer
            return message['content']

# Function to check if the assistant's response is adequate
def is_response_complete(query, response_text):
    check_prompt = f"""
    Determine if the following assistant's response adequately answers the user's query.

    User Query:
    {query}

    Assistant Response:
    {response_text}

    Respond with a JSON object: {{"completed": true}} if the response is adequate, or {{"completed": false}} if not.
    """
    check_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are to determine if the assistant's response adequately answers the user's query."},
            {"role": "user", "content": check_prompt}
        ]
    )
    try:
        completion_flag = json.loads(check_response['choices'][0]['message']['content'])
        return completion_flag.get("completed", False)
    except json.JSONDecodeError:
        return False

def query_pharma_bot_until_complete(query):
    while True:
        assistant_response = pharma_bot_assistant(query)
        st.write("### Assistant Response:")
        st.write(assistant_response)

        if is_response_complete(query, assistant_response):
            st.success("The assistant has adequately answered the query.")
            break
        else:
            st.warning("The assistant's response was not adequate. Refining query and trying again.")
            # Optionally, modify the query or provide more context
            # For simplicity, we'll just retry the same query
            continue

# Streamlit App Interface
st.title("PharmaBot Assistant")

st.write("Ask PharmaBot any questions about medications, their uses, side effects, interactions, dosage, and more.")

# User input
query = st.text_input("Enter your question about pharmaceutical products:")

if st.button("Ask PharmaBot"):
    with st.spinner("PharmaBot is thinking..."):
        query_pharma_bot_until_complete(query)

        
