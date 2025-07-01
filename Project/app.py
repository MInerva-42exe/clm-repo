import os
import re
import json
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from sqlalchemy import create_engine, text

app = Flask(__name__)

# Get the database connection string from environment variables
# This is the secret key you got from Neon
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Please set it before running the app.")

# Create a database engine
engine = create_engine(DATABASE_URL)

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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please set it before running the app.")
genai.configure(api_key=GEMINI_API_KEY)

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    # The 'engine' manages connections, so we just use it directly
    return engine.connect()

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
        # Clean the response to ensure it's valid JSON
        cleaned_text = response.text.strip().replace('```json', '').replace('```', '')
        extracted_info = json.loads(cleaned_text)
        return extracted_info
    except Exception as e:
        print(f"Error calling Gemini API or parsing response: {e}")
        return {"product": None, "document_type": None, "keywords": natural_language_query.lower().split()}

def build_sql_query_from_llm_output(extracted_info):
    """Builds a SQL WHERE clause for PostgreSQL."""
    conditions = []
    params = {}
    product = extracted_info.get("product")
    doc_type = extracted_info.get("document_type")
    keywords = extracted_info.get("keywords", [])

    if product:
        product_conditions = ['"Product" ILIKE :product']
        params['product'] = f"%{product}%"
        acronyms = PRODUCT_ACRONYM_MAP.get(product, [])
        for i, acronym in enumerate(acronyms):
            param_name = f"acronym_{i}"
            product_conditions.append(f'"Product" ILIKE :{param_name}')
            params[param_name] = f"%{acronym}%"
        conditions.append(f"({ ' OR '.join(product_conditions) })")

    if doc_type:
        doc_type_keyword_condition = '("Content_Title" ILIKE :doc_type OR "Description" ILIKE :doc_type OR "Generated_Keywords" ILIKE :doc_type)'
        conditions.append(f'("Doc_type" ILIKE :doc_type OR {doc_type_keyword_condition})')
        params['doc_type'] = f"%{doc_type}%"

    for i, keyword in enumerate(keywords):
        param_name = f"keyword_{i}"
        conditions.append(f'("Content_Title" ILIKE :{param_name} OR "Description" ILIKE :{param_name} OR "Generated_Keywords" ILIKE :{param_name})')
        params[param_name] = f"%{keyword}%"

    if not conditions:
        return "", {}
    
    # Note the double quotes around column names for PostgreSQL
    sql_where_clause = " AND ".join(conditions)
    return sql_where_clause, params

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handles the search query, queries the DB, and returns results."""
    user_query = request.json.get('query', '')
    
    extracted_info = llm_query_parsing(user_query)
    sql_where_clause, params = build_sql_query_from_llm_output(extracted_info)
    
    results = []
    if sql_where_clause:
        try:
            with get_db_connection() as conn:
                # Note the double quotes for column names
                query_string = f'SELECT "Product", "Doc_type", "Content_Title", "Description", "Link" FROM content_repo WHERE {sql_where_clause}'
                query = text(query_string)
                cursor = conn.execute(query, params)
                # Convert results to a list of dictionaries
                results = [dict(row._mapping) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Database query error: {e}")
            return jsonify({"error": "An error occurred during the database query."}), 500

    return jsonify(results)

@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarizes a single document from a given URL."""
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'summary': 'No URL provided.'}), 400

    summary = ""
    if 'manageengine.com' in url:
        summary = fetch_and_summarize_document(url)
    elif 'workdrive' in url:
        summary = "This is an internal document and cannot be summarized."
    else:
        summary = "This link is not from a recognized domain for summarization."
        
    return jsonify({'summary': summary})

# NOTE: The if __name__ == '__main__': block is removed as it's not needed for Gunicorn deployment
