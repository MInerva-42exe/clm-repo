import os
import json
import re
import requests
from bs4 import BeautifulSoup
import fitz
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# --- Database & AI Setup ---
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL: raise ValueError("DATABASE_URL is not set in .env file.")
engine = create_engine(DATABASE_URL)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY is not set in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

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
    if not conditions: return []
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
    print(f"--- FETCHING & SUMMARIZING ---")
    print(f"URL: {url}")
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
                for page in pdf_doc: page_text += page.get_text() + " "
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header']): element.decompose()
            page_text = soup.get_text(separator=' ', strip=True)
        if not page_text.strip(): return "Could not extract meaningful text."
        model = genai.GenerativeModel('gemini-1.5-flash')
        summarization_prompt = f"Please provide a concise, 2-3 sentence summary of the following document content:\n\n{page_text[:8000]}"
        summary_response = model.generate_content(summarization_prompt)
        return summary_response.text.strip()
    except Exception as e:
        return f"An error occurred during processing: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    # --- MEMORY: Get the history from the request ---
    history = request.json.get('history', [])
    if not user_message: return jsonify({"error": "No message provided."}), 400

    # --- MEMORY: Build the history string for the prompt ---
    history_string = ""
    for turn in history:
        # Sanitize content for the prompt
        role = "User" if turn['role'] == 'user' else "Assistant"
        content = turn['content'].split('\n')[0] 
        history_string += f"{role}: {content}\n"

    agent_prompt = f"""
You are WSM Content Assistant, a friendly, conversational, and highly intelligent AI expert on software documentation.
Your primary goal is to help users find documents by calling your `search_database` tool.
You MUST use the conversation history to understand the context of the user's latest message. For example, if the user first asks for "case studies for Product A" and then says "what about Product B", you must understand they are still asking for case studies and search for case studies for Product B. The history provides the context for the document type.

**CRITICAL INSTRUCTION: Mapping User Requests to Document Types**
You MUST intelligently map user requests to one of the following known `document_type` values: "Brochure or flyer", "Datasheet", "Presentation", "Technical Document", "Case study", "E-book or guide", "Solution brief", "Video", "Comparison document", "ROI calculator", "Other".

**YOUR RESPONSE LOGIC:**
1.  **Tool Call:** Based on the user's message AND the conversation history, if you have enough information to search, respond ONLY with a valid JSON object to call the `search_database` tool.
2.  **Clarification:** If the request is too vague even with the history, ask clarifying questions.
3.  **Conversation:** If the user is just chatting, respond naturally.

**Conversation History:**
{history_string}
**User's latest message:** "{user_message}"
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(agent_prompt)
        ai_response_text = response.text.strip()
        
        json_match = re.search(r'\{.*\}', ai_response_text, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
            try:
                tool_call = json.loads(json_string)
                if "document_type" in tool_call or "product" in tool_call or "keywords" in tool_call:
                    documents = search_database(
                        product=tool_call.get("product"),
                        document_type=tool_call.get("document_type"),
                        keywords=tool_call.get("keywords")
                    )
                    response_message = f"I found {len(documents)} document(s) for you:" if documents else "I couldn't find any documents that match your request. Please try different terms."
                    return jsonify({"type": "documents", "message": response_message, "data": documents})
            except json.JSONDecodeError: pass
        
        return jsonify({"type": "conversation", "message": ai_response_text})

    except Exception as e:
        print(f"An error occurred in the chat endpoint: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    url = request.json.get('url')
    if not url: return jsonify({'summary': 'No URL provided.'}), 400
    summary = fetch_and_summarize_document(url)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
