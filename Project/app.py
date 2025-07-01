import os
import re
import json
import sqlite3
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# The database path will be provided by an environment variable on the server.
# We'll use a default for local development.
DATABASE = os.environ.get('DATABASE_PATH', 'master.db')

# Mapping products to their acronyms for two-way search
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
    "Log360": [],
}

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDduuVQC0X8ugn8Srqhv7t5go5z6UEsVds")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please set it before running the app.")
genai.configure(api_key=GEMINI_API_KEY)

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_and_summarize_document(url):
    """Fetches content from a URL, extracts text from HTML or PDF, and summarizes it."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()

        page_text = ""
        content_type = response.headers.get('Content-Type', '')

        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            pdf_document = fitz.open(stream=response.content, filetype="pdf")
            for page in pdf_document:
                page_text += page.get_text()
            pdf_document.close()
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'div', 'article']):
                 page_text += element.get_text(separator=' ', strip=True) + " "

        if not page_text.strip():
            return "Could not extract meaningful text from the document."

        model = genai.GenerativeModel('gemini-1.5-flash')
        summarization_prompt = f"Please summarize the following document content in 2-3 sentences:\n\n{page_text[:8000]}"
        summary_response = model.generate_content(summarization_prompt)
        
        return summary_response.text.strip()

    except requests.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An error occurred during processing: {e}"


def llm_query_parsing(natural_language_query):
    """Uses Gemini to parse a natural language query into structured JSON."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
You are a sophisticated query-parsing assistant for a software documentation database. Your primary goal is to translate a user's natural language query into a structured JSON object. You must be precise and use reasoning to handle synonyms, acronyms, and plurals.

**Your Task:**
Analyze the user's query and extract the 'product', 'document_type', and 'keywords'.

**1. Product Identification:**
You MUST map the user's input to one of the following canonical product names. Be vigilant for acronyms and common variations.
- "ADManager Plus" (Acronyms: ADMP)
- "ADAudit Plus" (Acronyms: ADAP)
- "ADSelfService Plus" (Acronyms: ADSSP)
- "Recovery Manager Plus" (Acronyms: RMP)
- "Exchange Reporter Plus" (Acronyms: ERP)
- "M365 Manager Plus" (Acronyms: MMP, M365MP)
- "SharePoint Manager Plus" (Acronyms: SPMP)
- "DataSecurity Plus" (Acronyms: DSP)
- "Log360"
- "Identity360" (Acronyms: ID360)
- "AD360"

**2. Document Type Identification:**
Use your language understanding to map the user's request to ONE of the following canonical document types. Handle plurals and synonyms intelligently (e.g., "brochures" -> "Brochure or flyer", "specs" -> "Datasheet").
- "Brochure or flyer"
- "Datasheet"
- "Presentation"
- "Technical Document"
- "Case study"
- "E-book or guide"
- "Solution brief"
- "Video"
- "Comparison document"
- "ROI calculator"
- "Other"

**3. Keyword Extraction:**
These are important nouns or verbs from the query that are NOT product names or document types.

**4. CRITICAL RULES:**
- If the user's query *exactly* matches a product name or one of its acronyms (case-insensitive), you MUST set the 'product' field and leave the 'keywords' list empty.
- Do NOT confuse a file format like "PDF" with a document type. If the user mentions a file format, treat it as a 'keyword'.

**JSON Output Structure:**
Return a single, valid JSON object.
{{
    "product": "string | null",
    "document_type": "string | null",
    "keywords": "list of strings"
}}

---
**User Query:** "{natural_language_query}"
**Output:**
"""
    try:
        response = model.generate_content(prompt)
        extracted_info = json.loads(response.text.strip())
        return extracted_info
    except Exception as e:
        print(f"Error calling Gemini API or parsing response: {e}")
        return {"product": None, "document_type": None, "keywords": natural_language_query.lower().split()}

def build_sql_query_from_llm_output(extracted_info):
    """Builds a SQL WHERE clause, now searching the new Generated_Keywords column."""
    conditions = []
    params = []
    product = extracted_info.get("product")
    doc_type = extracted_info.get("document_type")
    keywords = extracted_info.get("keywords", [])

    if product:
        product_conditions = ["Product LIKE ?"]
        params.append(f"%{product}%")
        acronyms = PRODUCT_ACRONYM_MAP.get(product, [])
        for acronym in acronyms:
            product_conditions.append("Product LIKE ?")
            params.append(f"%{acronym}%")
        conditions.append(f"({ ' OR '.join(product_conditions) })")

    if doc_type:
        doc_type_keyword_condition = "(Content_Title LIKE ? OR Description LIKE ? OR Generated_Keywords LIKE ?)"
        conditions.append(f"(Doc_type LIKE ? OR {doc_type_keyword_condition})")
        params.extend([f"%{doc_type}%", f"%{doc_type}%", f"%{doc_type}%", f"%{doc_type}%"])
    
    for keyword in keywords:
        # Now searches the new Generated_Keywords column as well
        conditions.append("(Content_Title LIKE ? OR Description LIKE ? OR Generated_Keywords LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])

    if not conditions:
        return "", []
    sql_where_clause = " AND ".join(conditions)
    return sql_where_clause, params

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handles the search query, queries the DB, and returns results without summarization."""
    user_query = request.json.get('query', '')
    
    extracted_info = llm_query_parsing(user_query)
    sql_where_clause, params = build_sql_query_from_llm_output(extracted_info)
    
    results = []
    if sql_where_clause:
        conn = get_db_connection()
        try:
            query = f"SELECT Product, Doc_type, Content_Title, Description, Link FROM content_repo WHERE {sql_where_clause}"
            cursor = conn.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Database query error: {e}")
            return jsonify({"error": "An error occurred during the database query."}), 500
        finally:
            conn.close()

    return jsonify(results)

@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarizes a single document from a given URL."""
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'summary': 'No URL provided.'}), 400

    summary = ""
    # **FIX:** Changed the check to be more general.
    if 'manageengine.com' in url:
        summary = fetch_and_summarize_document(url)
    elif 'workdrive' in url:
        summary = "This is an internal document and cannot be summarized."
    else:
        # This case might not be needed if all links are one of the above.
        summary = "This link is not from a recognized domain for summarization."
        
    return jsonify({'summary': summary})

