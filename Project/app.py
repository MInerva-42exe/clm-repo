import os
import re
import json
import sqlite3
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

app = Flask(__name__)

# --- Database & AI Setup ---
# For Render, this will be set in the environment. For local, it's in .env
DATABASE_FILE = os.environ.get('DATABASE_FILE_PATH', 'master.db')

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
genai.configure(api_key=GEMINI_API_KEY)


PRODUCT_ACRONYM_MAP = {
    "ADManager Plus": ["ADMP"], "ADAudit Plus": ["ADAP"], "ADSelfService Plus": ["ADSSP"],
    "Recovery Manager Plus": ["RMP"], "Exchange Reporter Plus": ["ERP"], "M365 Manager Plus": ["MMP", "M365MP"],
    "SharePoint Manager Plus": ["SPMP"], "DataSecurity Plus": ["DSP"], "Identity360": ["ID360"],
    "AD360": [], "Log360": [],
}

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_and_summarize_document(url):
    """Fetches and summarizes a document from a URL."""
    try:
        if 'workdrive' in url:
            return "This is an internal document and cannot be summarized."
        
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
            return "Could not extract meaningful text from the document."

        model = genai.GenerativeModel('gemini-1.5-flash')
        summarization_prompt = f"Please provide a concise, 2-3 sentence summary of the following document content:\n\n{page_text[:8000]}"
        summary_response = model.generate_content(summarization_prompt)
        return summary_response.text.strip()

    except Exception as e:
        return f"An error occurred during processing: {e}"

def llm_query_parsing(natural_language_query):
    """Uses Gemini to parse a query into a structured JSON object."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
You are a highly efficient query-parsing AI. Your sole job is to convert a user's search query into a structured JSON object. You must be accurate and functional.

**Your Task:**
Analyze the user's query. Extract the 'product', 'document_type', and any 'keywords'.

- **product**: The specific ManageEngine product name.
- **document_type**: The category of the document.
- **keywords**: Any other important search terms.

**CRITICAL RULES:**
- You MUST respond with ONLY a valid JSON object. Do not add any conversational text, greetings, or explanations.
- If the user's query exactly matches a product name or acronym, set the 'product' field and leave keywords empty.
- Map user terms to the canonical types below (e.g., "flyer" -> "Brochure or flyer", "specs" -> "Datasheet").

**Canonical Document Types:**
"Brochure or flyer", "Datasheet", "Presentation", "Technical Document", "Case study", "E-book or guide", "Solution brief", "Video", "Comparison document", "ROI calculator", "Other"

---
**User Query:** "{natural_language_query}"

**JSON Output:**
"""
    try:
        response = model.generate_content(prompt)
        json_text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        return json.loads(json_text)
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        return {"keywords": natural_language_query.lower().split()}

def build_sql_query(extracted_info):
    """Builds a SQL WHERE clause from the structured info."""
    conditions, params = [], []
    product = extracted_info.get("product")
    doc_type = extracted_info.get("document_type")
    keywords = extracted_info.get("keywords", [])

    if product:
        product_conditions = ["Product LIKE ?"]
        params.append(f"%{product}%")
        for acronym in PRODUCT_ACRONYM_MAP.get(product, []):
            product_conditions.append("Product LIKE ?")
            params.append(f"%{acronym}%")
        conditions.append(f"({ ' OR '.join(product_conditions) })")

    if doc_type:
        doc_type_search = f"(Content_Title LIKE ? OR Description LIKE ? OR Generated_Keywords LIKE ?)"
        conditions.append(f"(Doc_type LIKE ? OR {doc_type_search})")
        params.extend([f"%{doc_type}%", f"%{doc_type}%", f"%{doc_type}%", f"%{doc_type}%"])
    
    for keyword in keywords:
        conditions.append(f"(Content_Title LIKE ? OR Description LIKE ? OR Generated_Keywords LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])

    if not conditions:
        return "", []
    return " AND ".join(conditions), params

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handles a search request."""
    user_query = request.json.get('query', '')
    if not user_query:
        return jsonify([])

    extracted_info = llm_query_parsing(user_query)
    sql_where_clause, params = build_sql_query(extracted_info)
    
    if not sql_where_clause:
        return jsonify([])

    try:
        conn = get_db_connection()
        query = f"SELECT Product, Doc_type, Content_Title, Description, Link FROM content_repo WHERE {sql_where_clause} LIMIT 20"
        cursor = conn.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(results)
    except Exception as e:
        print(f"Database query error: {e}")
        return jsonify({"error": "A database error occurred."}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarizes a single document."""
    url = request.json.get('url')
    if not url:
        return jsonify({'summary': 'No URL provided.'}), 400
    
    summary = fetch_and_summarize_document(url)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    # The app will run on the port defined by the environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
