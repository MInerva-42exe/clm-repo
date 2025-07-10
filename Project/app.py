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

# 1. Load keys from the comma-separated string into a list
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_STR.split(',')]
if not GEMINI_API_KEYS:
    raise ValueError("No API keys found in GEMINI_API_KEYS environment variable.")

# 2. Create a cycler that will loop through the keys endlessly
api_key_cycler = cycle(GEMINI_API_KEYS)

# 3. Configure the genai client with the first key from the list
try:
    initial_key = next(api_key_cycler)
    genai.configure(api_key=initial_key)
    print(f"Configured with initial API key ending in '...{initial_key[-4:]}'")
except StopIteration:
    raise ValueError("The API key list is empty.")


# --- Wrapper function for resilient API calls ---
def generate_content_with_failover(*args, **kwargs):
    """
    A wrapper for model.generate_content that automatically switches API keys
    on permission or quota-related failures.
    """
    keys_to_try = len(GEMINI_API_KEYS)
    for _ in range(keys_to_try):
        try:
            # Create a new model instance for the current configuration
            model = genai.GenerativeModel('gemini-1.5-flash')
            # Attempt the API call
            return model.generate_content(*args, **kwargs)
        except (google.api_core.exceptions.PermissionDenied, google.api_core.exceptions.ResourceExhausted) as e:
            print(f"API key failed with error: {e}. Trying next key.")
            
            # Get the next key from our cycler
            new_key = next(api_key_cycler)
            print(f"Switching to new API key ending in '...{new_key[-4:]}'")
            
            # Reconfigure the genai client with the new key
            genai.configure(api_key=new_key)
            
            # The loop will now retry with the new key configured on the next iteration
            continue
    
    # If the loop completes without a successful call, all keys have failed.
    raise Exception("All available API keys failed. Please check your keys and quotas.")


# --- Product Acronym Mapping ---
PRODUCT_ACRONYM_MAP = {
    "ADManager Plus": ["ADMP"],
    "ADAudit Plus": ["ADAP"],
    "ADSelfService Plus": ["ADSSP"],
    "Recovery Manager Plus": ["RMP"],
    "Exchange Reporter Plus": ["ERP"],
    "M365 Manager Plus": ["MMP", "M365MP"],
    "SharePoint Manager Plus": ["SPMP"],
    "DataSecurity Plus": ["DSP"],
    "Identity360": ["ID360"],
    "AD360": [],
    "Log360": []
}

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
    try:
        if 'workdrive' in url:
            return "This is an internal document and cannot be summarized automatically."
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=20, headers=headers, allow_redirects=True)
        response.raise_for_status()
        page_text = ""
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            with fitz.open(stream=response.content, filetype="pdf") as pdf_doc:
                for page in pdf_doc:
                    page_text += page.get_text() + " "
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            page_text = soup.get_text(separator=' ', strip=True)
        if not page_text.strip():
            return "Could not extract meaningful text."
        
        summarization_prompt = f"Please provide a concise, 2-3 sentence summary of the following document content:\n\n{page_text[:8000]}"
        
        # Use the failover wrapper for the API call
        summary_response = generate_content_with_failover(summarization_prompt)
        
        return summary_response.text.strip()
    except Exception as e:
        return f"An error occurred during processing: {e}"


# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    # This is the non-streaming version of the chat route
    # It combines the failover logic with your original tool-calling logic
    try:
        # This prompt is simplified. You would re-integrate your full tool-calling prompt here.
        agent_prompt = f"""
        You are WSM Content Assistant. Analyze the user's message and determine if you should search the database.
        User's message: "{user_message}"
        """
        
        # Use the failover wrapper for the API call
        response = generate_content_with_failover(agent_prompt)
        ai_response_text = response.text.strip()
        
        # You would add your logic here to parse the response and call search_database
        # For simplicity, this example just returns the conversational text.
        
        return jsonify({"type": "conversation", "message": ai_response_text})

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
