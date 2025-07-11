import os
import json
import logging
from itertools import cycle
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
app = Flask(
    __name__,
    static_folder="static",       # serve style.css (and other assets)
    template_folder="templates"   # render index.html
)
CORS(app, resources={
    r"/chat":       {"origins": "https://clm-repo.onrender.com"},
    r"/summarize":  {"origins": "https://clm-repo.onrender.com"},
    r"/health":     {"origins": "https://clm-repo.onrender.com"}
})
logging.basicConfig(level=logging.INFO)

# --- Database & AI Key Setup ---
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be set in environment")
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
    raise RuntimeError("GEMINI_API_KEYS must be set in environment")
GEMINI_API_KEYS = [k.strip() for k in GEMINI_API_KEYS_STR.split(",") if k.strip()]
if not GEMINI_API_KEYS:
    raise RuntimeError("No valid Gemini API keys found.")
api_key_cycler = cycle(GEMINI_API_KEYS)
genai.configure(api_key=next(api_key_cycler))

def generate_content_with_failover(*args, **kwargs):
    for _ in GEMINI_API_KEYS:
        try:
            model_kwargs = {k: v for k, v in kwargs.items()
                            if k in ("tools", "system_instruction", "generation_config")}
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                **model_kwargs
            )
            return model.generate_content(*args)
        except (google.api_core.exceptions.PermissionDenied,
                google.api_core.exceptions.ResourceExhausted) as e:
            app.logger.warning(f"Gemini key failed: {e}. Rotating key.")
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
Your sole job when users ask for documents is to call the `search_database` tool; otherwise, respond naturally.

Product map: {json.dumps(PRODUCT_ACRONYM_MAP)}
Doc types: {json.dumps(VALID_DOC_TYPES)}
"""

# --- Semantic Search ---
@lru_cache(maxsize=128)
def _cached_search(product, document_type, keyword_tuple):
    return tuple(_search_database(product, document_type, list(keyword_tuple)))

def search_database(product=None, document_type=None, keywords=None):
    return list(_cached_search(product or "", document_type or "", tuple(keywords or [])))

def _search_database(product: str = "", document_type: str = "", keywords: list = None):
    keywords = keywords or []
    query_text = " ".join(keywords).strip() or f"{product} {document_type}".strip()
    if not query_text:
        return []

    try:
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=query_text,
            task_type="retrieval_query",
            output_dimensionality=384
        )["embedding"]
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
            return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        app.logger.error(f"Database vector search error: {e}", exc_info=True)
        return []

# --- Document Summarization ---
def fetch_and_summarize_document(url):
    app.logger.info(f"Attempting to summarize URL: {url}")
    if 'workdrive' in url:
        return "This is an internal document and cannot be summarized."

    allowed_domain = 'download.manageengine.com' in url or 'manageengine.com' in url or url.lower().endswith('.pdf')
    if not allowed_domain:
        return "This document cannot be summarized."

    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type','').lower()
        text = ""
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            with fitz.open(stream=resp.content, filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text() + " "
        else:
            soup = BeautifulSoup(resp.content, 'html.parser')
            for tag in soup(['script','style','nav','footer','header']):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)

        if not text.strip():
            return "Could not extract meaningful text from the document."

        prompt = f"Please provide a concise, 2-3 sentence summary of the following content:\n\n{text[:8000]}"
        resp_ai = generate_content_with_failover(
            [{'role':'user','parts':[{'text':prompt}]}],
            system_instruction="You are a text summarizer.",
            generation_config=genai.types.GenerationConfig(temperature=0.4)
        )
        return resp_ai.candidates[0].content.parts[0].text
    except Exception as e:
        app.logger.error(f"Error during summarization: {e}", exc_info=True)
        return "An error occurred while trying to summarize the document."

# --- Routes ---
@app.route("/", methods=["GET"])
def serve_frontend():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
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
    try:
        ai_resp = generate_content_with_failover(
            conversation,
            tools=[search_tool],
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        part = ai_resp.candidates[0].content.parts[0]
        if getattr(part, "function_call", None) and part.function_call.name == "search_database":
            docs = search_database(**part.function_call.args)
            if docs:
                return jsonify({"type":"documents","message":f"I found {len(docs)} document(s):","data":docs})
            return jsonify({"type":"conversation","message":"I couldn't find any documents that match your request."})
        return jsonify({"type":"conversation","message":getattr(part, "text", "I’m not sure how to respond.")})
    except Exception as e:
        app.logger.error(f"Error in /chat: {e}", exc_info=True)
        return jsonify({"error":"An error occurred while processing your request."}), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json() or {}
    url = data.get("url")
    if not url:
        return jsonify({"status":"error","message":"No URL provided."}), 200

    summary = fetch_and_summarize_document(url)
    if summary.startswith("An error") or "cannot" in summary:
        return jsonify({"status":"error","message":summary}), 200
    return jsonify({"status":"success","summary":summary})

if __name__ == "__main__":
    # bind to all interfaces and honor Render's PORT
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
