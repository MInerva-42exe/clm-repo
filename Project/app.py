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
from redis import Redis
# --- UPDATED: Added 'Job' to the import line ---
from rq import Queue, get_current_job, Retry, Job
import requests
from bs4 import BeautifulSoup
import fitz # PyMuPDF

# --- Setup ---
load_dotenv()
app = Flask(__name__)
# Remember to replace "*" with your frontend's exact origin for production
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)

# --- Redis Queue Setup ---
REDIS_URL = os.environ.get('REDIS_URL')
if not REDIS_URL:
    raise RuntimeError("REDIS_URL is not configured.")
redis_conn = Redis.from_url(REDIS_URL)
q = Queue(connection=redis_conn)

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
api_key_cycler = cycle(GEMINI_API_KEYS)
genai.configure(api_key=next(api_key_cycler))


# --- Product & Document Type Definitions ---
PRODUCT_ACRONYM_MAP = {
    "ADManager Plus": ["ADMP"], "ADAudit Plus": ["ADAP"], "ADSelfService Plus": ["ADSSP"],
    "Recovery Manager Plus": ["RMP"], "Exchange Reporter Plus": ["ERP"], "M365 Manager Plus": ["MMP", "M365MP"],
    "SharePoint Manager Plus": ["SPMP"], "DataSecurity Plus": ["DSP"], "Identity360": ["ID360"],
    "AD360": ["AD360", "AD 360", "ManageEngine AD360"],
    "Log360": ["Log360", "Log 360", "ManageEngine Log360"]
}
VALID_DOC_TYPES = [
    "Brochure or flyer", "Datasheet", "Presentation", "Technical Document",
    "Case study", "E-book or guide", "Solution brief", "Video",
    "Comparison document", "ROI calculator", "Other"
]

# --- Normalization Helpers ---
def normalize_product(product: str) -> str:
    """Map any acronym or variant back to the canonical full product name."""
    product = product.strip()
    for full_name, acronyms in PRODUCT_ACRONYM_MAP.items():
        if product.lower() == full_name.lower() or product in acronyms:
            return full_name
    return product


def normalize_document_type(doc_type: str) -> str:
    """Map incoming doc_type to one of the VALID_DOC_TYPES if it matches."""
    doc_type = doc_type.strip().lower()
    for valid in VALID_DOC_TYPES:
        if doc_type == valid.lower() or doc_type in valid.lower():
            return valid
    return doc_type

# --- Tool Definition & System Prompt ---
search_tool = {
    "function_declarations": [{
        "name": "search_database",
        "description": "Searches the database for documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "product": {"type": "string"},
                "document_type": {"type": "string"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": []
        }
    }]
}

SYSTEM_PROMPT = f"""
You are WSM Content Assistant, a function-calling AI that helps users find documents.
Your job is to analyze user queries and call the `search_database` tool with the appropriate parameters.
If the user is making small talk, respond conversationally. If a query is too vague, ask for clarification.

- Product Map: {json.dumps(PRODUCT_ACRONYM_MAP)}
- Document Types: {json.dumps(VALID_DOC_TYPES)}
"""

# --- Functions for the Worker ---
def _update_job_progress(message: str):
    job = get_current_job()
    if job:
        job.meta['progress'] = message
        job.save_meta()

@lru_cache(maxsize=128)
def _cached_search(product, document_type, keyword_tuple):
    results = _search_database(product, document_type, list(keyword_tuple))
    return tuple(results) if results is not None else tuple()


def search_database(product=None, document_type=None, keywords=None):
    """Public-facing search function that normalizes inputs, then uses the cache."""
    normalized_product = normalize_product(product or "")
    normalized_doc_type = normalize_document_type(document_type or "")
    key = (
        normalized_product,
        normalized_doc_type,
        tuple(sorted(keywords or []))
    )
    return list(_cached_search(*key))


def _search_database(product: str = "", document_type: str = "", keywords: list = None):
    """Searches the database using vector similarity and smart filters."""
    app.logger.info(f"DATABASE SEARCH: Product='{product}', Type='{document_type}', Keywords={keywords}")
    keywords = keywords or []
    query_text = " ".join(keywords).strip() or f"{product} {document_type}".strip()
    if not query_text:
        return []

    try:
        resp = genai.embed_content(
            model="models/embedding-001",
            content=query_text,
            task_type="retrieval_query"
        )
        embedding = resp['embedding']
    except Exception as e:
        app.logger.error(f"Failed to embed search query: {e}")
        return []

    base_sql = """
        SELECT "Product","Doc_type","Content_Title","Description","Link",
               1 - (embedding <=> :emb) AS similarity
         FROM content_repo
        WHERE embedding IS NOT NULL
    """
    params = {"emb": json.dumps(embedding)}
    filters = []
    if product:
        variants = [product] + PRODUCT_ACRONYM_MAP.get(product, [])
        params["pv"] = [f"%{v}%" for v in variants]
        filters.append('"Product" ILIKE ANY(:pv)')
    if document_type:
        params["dt"] = f"%{document_type}%"
        filters.append('"Doc_type" ILIKE :dt')
    if filters:
        base_sql += " AND " + " AND ".join(filters)
    base_sql += " ORDER BY similarity DESC LIMIT 10"

    try:
        with engine.connect() as conn:
            result = conn.execute(text(base_sql), params)
            rows = result.fetchall()
            app.logger.info(f"Found {len(rows)} results from database.")
            return [dict(row._mapping) for row in rows]
    except Exception as e:
        app.logger.error(f"Database vector search error: {e}", exc_info=True)
        return []

def call_generative_model(conversation_history):
    """This function is executed by the background worker for chat."""
    app.logger.info("WORKER: Received chat job.")
    try:
        _update_job_progress("Analyzing request...")
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=SYSTEM_PROMPT, tools=[search_tool])
        response = model.generate_content(conversation_history, generation_config=genai.types.GenerationConfig(temperature=0.1))
        part = response.candidates[0].content.parts[0]

        if getattr(part, "function_call", None):
            _update_job_progress("Searching database...")
            docs = search_database(**part.function_call.args)
            if docs:
                return {"type": "documents", "message": f"I found {len(docs)} document(s):", "data": docs}
            return {"type": "conversation", "message": "I searched, but couldn't find any documents that match your request."}
        
        _update_job_progress("Formatting response...")
        return {"type": "conversation", "message": getattr(part, "text", "I’m not sure how to respond.")}
    except Exception as e:
        app.logger.error(f"WORKER ERROR in call_generative_model: {e}", exc_info=True)
        return {"error": "An error occurred while analyzing your request."}

def call_summarize(url):
    """This function is executed by the background worker for summarization."""
    app.logger.info(f"WORKER: Starting summarization for {url}")
    _update_job_progress("Fetching document content...")

    if 'workdrive' in url:
        return {"status": "error", "message": "This is an internal document and cannot be summarized."}

    allowed_domain = 'download.manageengine.com' in url or 'manageengine.com' in url or url.lower().endswith('.pdf')
    if not allowed_domain:
        return {"status": "error", "message": "This document cannot be summarized."}

    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type','').lower()
        page_text = ""
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            with fitz.open(stream=resp.content, filetype="pdf") as pdf:
                for page in pdf:
                    page_text += page.get_text() + " "
        else:
            soup = BeautifulSoup(resp.content, 'html.parser')
            for tag in soup(['script','style','nav','footer','header']):
                tag.decompose()
            page_text = soup.get_text(separator=' ', strip=True)

        if not page_text.strip():
            return {"status": "error", "message": "Could not extract meaningful text."}

        _update_job_progress("Summarizing content...")
        prompt = f"""Please provide a concise, 2-3 sentence summary of the following content:

{page_text[:8000]}"""
        model = genai.GenerativeModel('gemini-1.5-flash')
        resp_ai = model.generate_content(
            [{'role':'user','parts':[{'text':prompt}]}],
            generation_config=genai.types.GenerationConfig(temperature=0.4)
        )
        
        summary_text = resp_ai.candidates[0].content.parts[0].text
        return {"status": "success", "summary": summary_text}
    except Exception as e:
        app.logger.error(f"Error during summarization worker: {e}", exc_info=True)
        return {"status": "error", "message": "An error occurred during summarization."}


# --- Routes ---
@app.route("/")
def serve_frontend():
    return render_template("index.html")

@app.route("/health")
def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"app":"up","db":True}), 200
    except:
        return jsonify({"app":"up","db":False}), 503

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = data.get("message","").strip()
    history = data.get("history", [])
    if not user_message:
        return jsonify({"error":"No message provided."}), 400

    conversation = (history + [{'role':'user','parts':[{'text':user_message}]}])[-4:]
    job = q.enqueue(call_generative_model, args=[conversation], retry=Retry(max=3), job_timeout='5m', result_ttl=600)
    return jsonify({'job_id': job.id})

@app.route('/result/<job_id>')
def get_job_result(job_id):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        if job.is_finished:
            return jsonify({'status':'finished','result':job.result})
        elif job.is_failed:
            return jsonify({'status':'failed'})
        else:
            return jsonify({'status':'pending','progress': job.meta.get('progress','Processing...')})
    except Exception as e:
        app.logger.error(f"/result error: {e}", exc_info=True)
        return jsonify({'status':'error','error':str(e)}), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json() or {}
    url = data.get("url")
    if not url:
        return jsonify({"error":"No URL provided."}), 400

    job = q.enqueue(call_summarize, args=[url], retry=Retry(max=1), job_timeout='5m', result_ttl=600)
    app.logger.info(f"Enqueued summarization job {job.id} for URL: {url}")
    return jsonify({'job_id': job.id})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
