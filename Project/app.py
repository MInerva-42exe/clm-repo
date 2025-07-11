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
from rq import Queue, get_current_job
from rq.job import Retry
import requests
from bs4 import BeautifulSoup
import fitz # PyMuPDF

# --- Setup ---
load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://clm-repo.onrender.com"}})
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
    "AD360": ["AD360", "AD 360"]
}
VALID_DOC_TYPES = [
    "Brochure or flyer", "Datasheet", "Presentation", "Technical Document", "Case study", "E-book or guide", "Solution brief", "Video", "Comparison document", "ROI calculator", "Other"
]

search_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(name='search_database', description="Searches the database for documents.",
            parameters=genai.protos.Schema(type=genai.protos.Type.OBJECT,properties={'product': genai.protos.Schema(type=genai.protos.Type.STRING),'document_type': genai.protos.Schema(type=genai.protos.Type.STRING),'keywords': genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING))}))
    ]
)

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
    return tuple(_search_database(product, document_type, list(keyword_tuple)))

def search_database(product=None, document_type=None, keywords=None):
    return list(_cached_search(product or "", document_type or "", tuple(keywords or [])))

def _search_database(product: str = "", document_type: str = "", keywords: list = None):
    # This is where your real database search logic goes.
    # For now, it's a placeholder.
    app.logger.info(f"WORKER DB SEARCH: Product='{product}', Type='{document_type}', Keywords={keywords}")
    return [{"Product": product, "Doc_type": document_type, "Content_Title": "Real Document From DB", "Link": "#"}]

def call_generative_model(conversation_history):
    """This function is executed by the background worker."""
    app.logger.info("WORKER: Received job to process conversation.")
    try:
        _update_job_progress("Step 1/2: Analyzing request...")
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=SYSTEM_PROMPT, tools=[search_tool])
        response = model.generate_content(conversation_history, generation_config=genai.types.GenerationConfig(temperature=0.1))
        part = response.candidates[0].content.parts[0]

        if getattr(part, "function_call", None):
            _update_job_progress("Step 2/2: Searching database...")
            docs = search_database(**part.function_call.args)
            if docs:
                return {"type": "documents", "message": f"I found {len(docs)} document(s):", "data": docs}
            return {"type": "conversation", "message": "I couldn't find any documents that match your request."}
        
        _update_job_progress("Formatting response...")
        return {"type": "conversation", "message": getattr(part, "text", "I’m not sure how to respond.")}
    except Exception as e:
        app.logger.error(f"WORKER ERROR: {e}", exc_info=True)
        raise

# --- Routes (Now very fast) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """This route is now extremely fast. It just creates a job."""
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    conversation = (history + [{'role': 'user', 'parts': [{'text': user_message}]}])[-4:]
    
    job = q.enqueue(
        call_generative_model,
        args=[conversation],
        retry=Retry(max=3),
        job_timeout='5m',
        result_ttl=600
    )
    
    app.logger.info(f"Enqueued job {job.id} for user message: '{user_message}'")
    return jsonify({'job_id': job.id})

@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """The frontend polls this endpoint to get the final result."""
    job = q.fetch_job(job_id)
    if job:
        if job.is_finished:
            return jsonify({'status': 'finished', 'result': job.result})
        elif job.is_failed:
            return jsonify({'status': 'failed'})
        else:
            return jsonify({'status': 'pending', 'progress': job.meta.get('progress', 'Processing...')})
    return jsonify({'status': 'not_found'}), 404

# --- Other Routes ---
@app.route("/health")
def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"app":"up","db":True}), 200
    except:
        return jsonify({"app":"up","db":False}), 503

# You would add your /summarize route logic here, also using the background worker pattern.
@app.route("/summarize", methods=["POST"])
def summarize():
    return jsonify({"status":"error", "message":"Summarize not implemented yet."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
