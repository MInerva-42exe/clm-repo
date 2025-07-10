import os
import json
from itertools import cycle
import logging
from functools import lru_cache

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import google.api_core.exceptions
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

# --- Setup ---
load_dotenv()
app = Flask(__name__)
CORS(app, resources={
    r"/chat": {"origins": "https://clm-repo.onrender.com"},
    r"/summarize": {"origins": "https://clm-repo.onrender.com"},
    r"/health": {"origins": "https://clm-repo.onrender.com"}
})
logging.basicConfig(level=logging.INFO)

# --- Database & AI Key Setup ---
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env file.")
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    app.logger.info("✅ Database connected successfully!")
except OperationalError as e:
    app.logger.error(f"❌ Database connection failed on startup: {e}", exc_info=True)
    raise

GEMINI_API_KEYS_STR = os.environ.get("GEMINI_API_KEYS")
if not GEMINI_API_KEYS_STR:
    raise ValueError("GEMINI_API_KEYS is not set in .env file.")
GEMINI_API_KEYS = [k.strip() for k in GEMINI_API_KEYS_STR.split(',') if k.strip()]
if not GEMINI_API_KEYS:
    raise ValueError("No valid Gemini API keys found.")
api_key_cycler = cycle(GEMINI_API_KEYS)
genai.configure(api_key=next(api_key_cycler))


def generate_content_with_failover(*args, **kwargs):
    for _ in range(len(GEMINI_API_KEYS)):
        try:
            model_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ("tools", "system_instruction", "generation_config")
            }
            app.logger.debug(f"Calling Gemini with model_kwargs: {list(model_kwargs.keys())}")

            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                **model_kwargs
            )
            return model.generate_content(*args)
        except (google.api_core.exceptions.PermissionDenied,
                google.api_core.exceptions.ResourceExhausted) as e:
            app.logger.warning(f"Key failed: {e}. Rotating key.")
            genai.configure(api_key=next(api_key_cycler))
    raise RuntimeError("All Gemini API keys failed.")


# --- Product & Document Type Definitions ---
PRODUCT_ACRONYM_MAP = {
    "ADManager Plus": ["ADMP"], "ADAudit Plus": ["ADAP"], "ADSelfService Plus": ["ADSSP"],
    "Recovery Manager Plus": ["RMP"], "Exchange Reporter Plus": ["ERP"], "M365 Manager Plus": ["MMP", "M365MP"],
    "SharePoint Manager Plus": ["SPMP"], "DataSecurity Plus": ["DSP"], "Identity360": ["ID360"],
    "AD360": [], "Log360": []
}
VALID_DOC_TYPES = [
    "Brochure or flyer", "Datasheet", "Presentation", "Technical Document",
    "Case study", "E-book or guide", "Solution brief", "Video",
    "Comparison document", "ROI calculator", "Other"
]

# --- Tool Definition & System Prompt ---
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
                    'keywords': genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING)
                    )
                }
            )
        )
    ]
)

SYSTEM_PROMPT = f"""
You are WSM Content Assistant, a friendly, conversational AI expert on software documentation.
Your sole job when users ask for documents is to call the `search_database` tool; otherwise, respond naturally or ask for clarification.

=== 1. INPUT PROCESSING ===
1. PRODUCT NORMALIZATION (highest priority)
   - Map any product name or acronym to its canonical form using:
     {json.dumps(PRODUCT_ACRONYM_MAP)}
2. DOCUMENT-TYPE MAPPING
   - Map user requests to one of:
     {json.dumps(VALID_DOC_TYPES)}
3. KEYWORD EXTRACTION
   - Pull out 2–5 core phrases that capture user intent.

=== 2. DECISION LOGIC ===
- **If the user is clearly asking for documents**, immediately call the `search_database` tool.
- **If details are missing or ambiguous**, ask a follow-up question.
- **Otherwise**, carry on the conversation as normal.
"""

# --- Semantic Search ---
@lru_cache(maxsize=128)
def _cached_search(product, document_type, keyword_tuple):
    return tuple(_search_database(product, document_type, list(keyword_tuple)))

def search_database(product=None, document_type=None, keywords=None):
    return list(_cached_search(product or "", document_type or "", tuple(keywords or [])))

def _search_database(product: str = None, document_type: str = None, keywords: list = None):
    app.logger.info(f"VECTOR SEARCH: Product='{product}', Type='{document_type}', Keywords={keywords}")
    query_text = " ".join(keywords or []).strip() or f"{product or ''} {document_type or ''}".strip()
    if not query_text:
        return []

    try:
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=query_text,
            task_type="retrieval_query",
            output_dimensionality=384
        )["embedding"]
    except Exception as e:
        app.logger.error(f"Failed to embed search query: {e}")
        return []

    base_sql = """
        SELECT "Product", "Doc_type", "Content_Title", "Description", "Link",
               1 - (embedding <=> :query_embedding) AS similarity
        FROM content_repo
        WHERE embedding IS NOT NULL
    """
    filters = []
    params = {"query_embedding": json.dumps(query_embedding)}

    if product:
        variants = [product] + PRODUCT_ACRONYM_MAP.get(product, [])
        params["product_variants"] = [f"%{v}%" for v in variants]
        filters.append('"Product" ILIKE ANY(:product_variants)')

    if document_type:
        filters.append('"Doc_type" ILIKE :doc_type')
        params["doc_type"] = f"%{document_type}%"

    if filters:
        base_sql += " AND " + " AND ".join(filters)
    base_sql += " ORDER BY similarity DESC LIMIT 10"

    try:
        with engine.connect() as conn:
            cursor = conn.execute(text(base_sql), params)
            return [dict(row._mapping) for row in cursor.fetchall()]
    except Exception as e:
        app.logger.error(f"Database vector search error: {e}", exc_info=True)
        return []

# --- Document Summarization ---
def fetch_and_summarize_document(url):
    app.logger.info(f"Attempting to summarize URL: {url}")
    if 'workdrive' in url:
        return "This is an internal document and cannot be summarized."

    is_allowed_domain = 'download.manageengine.com' in url or 'manageengine.com' in url
    is_pdf = url.lower().endswith('.pdf')

    if is_allowed_domain or is_pdf:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, timeout=20, headers=headers, allow_redirects=True)
            response.raise_for_status()

            page_text = ""
            content_type = response.headers.get('Content-Type', '').lower()

            if 'application/pdf' in content_type or is_pdf:
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

            summarization_prompt = f"Please provide a concise, 2-3 sentence summary of the following document content:\n\n{page_text[:8000]}"
            summary_resp = generate_content_with_failover(
                [{'role': 'user', 'parts': [{'text': summarization_prompt}]}],
                system_instruction="You are a text summarizer.",
                generation_config=genai.types.GenerationConfig(temperature=0.4)
            )
            return summary_resp.candidates[0].content.parts[0].text
        except Exception as e:
            app.logger.error(f"Error during summarization process: {e}", exc_info=True)
            return "An error occurred while trying to summarize the document."
    else:
        return "This document cannot be summarized."

# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "WSM Content Assistant API is running.",
        "endpoints": {
            "POST /chat": "Send a message to the assistant",
            "POST /summarize": "Summarize a document",
            "GET /health": "Health check"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"app": "up", "db": True}), 200
    except Exception:
        return jsonify({"app": "up", "db": False}), 503

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_message = data.get('message', '').strip()
    history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        conversation_history = (history + [{'role': 'user', 'parts': [{'text': user_message}]}])[-4:]
        response = generate_content_with_failover(
            conversation_history,
            tools=[search_tool],
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )

        part = response.candidates[0].content.parts[0]
        if getattr(part, 'function_call', None) and part.function_call.name == "search_database":
            documents = search_database(**part.function_call.args)
            if documents:
                return jsonify({"type": "documents", "message": f"I found {len(documents)} document(s) for you:", "data": documents})
            return jsonify({"type": "conversation", "message": "I couldn't find any documents that match your request."})
        return jsonify({"type": "conversation", "message": getattr(part, 'text', 'I’m not sure how to respond.')})

    except Exception as e:
        app.logger.error(f"Error in /chat: {e}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request."}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json or {}
    url = data.get('url')
    if not url:
        return jsonify({'status': 'error', 'message': 'No URL provided.'}), 200

    summary = fetch_and_summarize_document(url)
    if "cannot be summarized" in summary or "An error occurred" in summary:
        return jsonify({'status': 'error', 'message': summary}), 200

    return jsonify({'status': 'success', 'summary': summary})


if __name__ == '__main__':
    app.run()
