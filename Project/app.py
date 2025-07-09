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
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env file.")
engine = create_engine(DATABASE_URL)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

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

# --- AI Tool Definition ---
# Define the database search function as a tool for the AI
search_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name='search_database',
            description="Searches the content database for documents based on product, document type, and keywords.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    'product': genai.protos.Schema(type=genai.protos.Type.STRING, description="The full product name, e.g., 'ADManager Plus'"),
                    'document_type': genai.protos.Schema(type=genai.protos.Type.STRING, description="The type of document, e.g., 'Case study' or 'Technical Document'"),
                    'keywords': genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="A list of keywords from the user's query."
                    )
                }
            )
        )
    ]
)

# Initialize the model with the tool
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    tools=[search_tool]
)

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
        summarization_model = genai.GenerativeModel('gemini-1.5-flash')
        summarization_prompt = f"Please provide a concise, 2-3 sentence summary of the following document content:\n\n{page_text[:8000]}"
        summary_response = summarization_model.generate_content(summarization_prompt)
        return summary_response.text.strip()
    except Exception as e:
        return f"An error occurred during processing: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    product_map_string = json.dumps(PRODUCT_ACRONYM_MAP, indent=2)
    agent_prompt = f"""
You are WSM Content Assistant, a friendly and helpful AI expert on software documentation. Your goal is to help users find documents.

**Instructions:**
1.  Analyze the user's message to identify a product, document type, and keywords.
2.  **Normalize Product Names:** You MUST normalize product names and acronyms using this map before calling the tool. For example, 'ADMP' should become 'ADManager Plus'.
    {product_map_string}
3.  **Call the Tool:** If you have enough information to search, call the `search_database` tool.
4.  **Converse:** If the user's message is not a search request (e.g., "hello", "thank you"), respond conversationally. If the request is ambiguous, ask clarifying questions.

User's message: "{user_message}"
"""
    try:
        response = model.generate_content(agent_prompt)
        response_part = response.candidates[0].content.parts[0]

        # Check if the model requested a tool call
        if response_part.function_call and response_part.function_call.name == "search_database":
            params = {key: value for key, value in response_part.function_call.args.items()}
            print(f"--- TOOL CALL DETECTED ---")
            print(f"Function: search_database, Parameters: {params}")

            documents = search_database(
                product=params.get("product"),
                document_type=params.get("document_type"),
                keywords=params.get("keywords")
            )

            response_message = f"I found {len(documents)} document(s) for you:" if documents else "I couldn't find any documents that match your request. Please try different terms."
            return jsonify({"type": "documents", "message": response_message, "data": documents})

        # If no tool call, it's a conversational response
        else:
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
