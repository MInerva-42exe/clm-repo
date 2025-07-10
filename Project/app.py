import os
import json
import re
import requests
from bs4 import BeautifulSoup
import fitz
from flask import Flask, render_template, request, jsonify, Response
import google.generativeai as genai
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from itertools import cycle
import google.api_core.exceptions

load_dotenv()
app = Flask(__name__)

# --- Database Setup ---
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env file.")
engine = create_engine(DATABASE_URL)


# --- API Key Manager Setup ---
GEMINI_API_KEYS_STR = os.environ.get("GEMINI_API_KEYS")
if not GEMINI_API_KEYS_STR:
    raise ValueError("GEMINI_API_KEYS is not set in .env file as a comma-separated string.")
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_STR.split(',')]
api_key_cycler = cycle(GEMINI_API_KEYS)
try:
    initial_key = next(api_key_cycler)
    genai.configure(api_key=initial_key)
    print(f"Configured with initial API key ending in '...{initial_key[-4:]}'")
except StopIteration:
    raise ValueError("The API key list is empty.")


# --- Wrapper function for resilient API calls ---
def generate_content_with_failover(*args, **kwargs):
    keys_to_try = len(GEMINI_API_KEYS)
    for _ in range(keys_to_try):
        try:
            model = genai.GenerativeModel(model_name='gemini-1.5-flash', tools=kwargs.pop('tools', None))
            return model.generate_content(*args, **kwargs)
        except (google.api_core.exceptions.PermissionDenied, google.api_core.exceptions.ResourceExhausted) as e:
            print(f"API key failed with error: {e}. Trying next key.")
            new_key = next(api_key_cycler)
            print(f"Switching to new API key ending in '...{new_key[-4:]}'")
            genai.configure(api_key=new_key)
            continue
    raise Exception("All available API keys failed. Please check your keys and quotas.")


# --- Database & Summarization Functions ---
def search_database(product: str = None, document_type: str = None, keywords: list = None):
    """The 'tool' for searching the database."""
    print(f"--- DATABASE SEARCH ---")
    print(f"Product: {product}, Doc Type: {document_type}, Keywords: {keywords}")
    conditions, params = [], {}
    if product:
        conditions.append('"Product" ILIKE :product')
        params['product'] = f"%{product}%"
    if document_type:
        conditions.append('"Doc_type" ILIKE :doc_type')
        params['doc_type'] = f"%{document_type}%"
    if keywords:
        keyword_conditions = []
        for i, keyword in enumerate(keywords):
            param_name = f"keyword_{i}"
            keyword_search_clause = (f'("Content_Title" ILIKE :{param_name} OR "Description" ILIKE :{param_name} OR "Generated_Keywords" ILIKE :{param_name})')
            keyword_conditions.append(keyword_search_clause)
            params[param_name] = f"%{keyword}%"
        if keyword_conditions:
            conditions.append(f"({ ' OR '.join(keyword_conditions) })")
    if not conditions:
        return []
    sql_where_clause = " AND ".join(conditions)
    try:
        with engine.connect() as conn:
            query_string = f'SELECT "Product", "Doc_type", "Content_Title", "Description", "Link" FROM content_repo WHERE {sql_where_clause} LIMIT 10'
            cursor = conn.execute(text(query_string), params)
            return [dict(row._mapping) for row in cursor.fetchall()]
    except Exception as e:
        print(f"Database query error: {e}")
        return []

def fetch_and_summarize_document(url):
    """Fetches content from a URL and summarizes it."""
    # ... (Your full summarization logic would go here)
    return "This is a summary of the document."


# --- Tool & Model Definition ---
search_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name='search_database',
            description="Searches the content database for documents.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    'product': genai.protos.Schema(type=genai.protos.Type.STRING),
                    'document_type': genai.protos.Schema(type=genai.protos.Type.STRING),
                    'keywords': genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING))
                }
            )
        )
    ]
)

# --- NEW: System prompt to guide the AI's behavior ---
SYSTEM_PROMPT = """
You are WSM Content Assistant, a friendly and helpful AI expert on software documentation.

Your primary function is to help users find documents using the `search_database` tool.

However, you are also a conversational AI. If the user's message is a simple greeting (like "hello", "hi"), a question about you ("who are you?"), or any other casual conversation, you MUST respond naturally and conversationally. Do NOT call the `search_database` tool for these types of messages.

Only call the `search_database` tool when the user explicitly asks to find a document or provides clear search terms (like a product name, document type, or keywords).
"""

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        # Pass both the system prompt and the user message to the model
        response = generate_content_with_failover(
            [SYSTEM_PROMPT, user_message],
            tools=[search_tool]
        )
        
        response_part = response.candidates[0].content.parts[0]

        if response_part.function_call.name == "search_database":
            params = {key: value for key, value in response_part.function_call.args.items()}
            documents = search_database(**params)
            
            if documents:
                return jsonify({"type": "documents", "message": f"I found {len(documents)} document(s) for you:", "data": documents})
            else:
                return jsonify({"type": "conversation", "message": "I couldn't find any documents that match your request. Please try different terms."})

        else:
            # If no tool was called, it's a conversational response
            return jsonify({"type": "conversation", "message": response.text})

    except Exception as e:
        print(f"An error occurred in the chat endpoint: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500


@app.route('/summarize', methods=['POST'])
def summarize():
    url = request.json.get('url')
    if not url:
        return jsonify({'summary': 'No URL provided.'}), 400
    summary = fetch_and_summarize_document(url)
    return jsonify({'summary': summary})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
