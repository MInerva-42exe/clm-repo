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
import fitz # PyMuPDF

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
     E.g. “Active Directory user provisioning” → ["active directory", "user provisioning"]

=== 2. DECISION LOGIC ===
- **If the user is clearly asking for documents**, immediately call the `search_database` tool.
- **If details are missing or ambiguous**, ask a follow-up question.
- **Otherwise**, carry on the conversation as normal.
"""


# --- Semantic Search Implementation ---
@lru_cache(maxsize=128)
def _cached_search(product, document_type, keyword_tuple):
    """Internal cached search function."""
    # UPDATED: Safer handling of the search result
    results = _search_database(product, document_type, list(keyword_tuple))
    if not results:
        app.logger.warning("'_search_database' returned None or empty. Returning empty tuple to prevent crash.")
        return tuple()
    return tuple(results)

def search_database(product=None, document_type=None, keywords=None):
    """Public-facing search function that uses the cache."""
    return list(_cached_search(product or "", document_type or "", tuple(keywords or [])))

def _search_database(product: str = "", document_type: str = "", keywords: list = None):
    """Searches the database using vector similarity and smart filters."""
    app.logger.info(f"VECTOR SEARCH: Product='{product}', Type='{document_type}', Keywords={keywords}")
    
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
            rows = result.fetchall()
            app.logger.info(f"Found {len(rows)} results from database.")
            return [dict(row._mapping) for row in rows]
    except Exception as e:
        # UPDATED: Return an empty list on error instead of raising
        app.logger.error(f"Database vector search error: {e}", exc_info=True)
        return []

# --- Document Summarization ---
def fetch_and_summarize_document(url):
    # ... (Your summarization logic remains here)
    pass

# --- Routes ---
@app.route("/", methods=["GET"])
def serve_frontend():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    # ... (Your health check logic remains here)
    pass

@app.route("/chat", methods=["POST"])
def chat():
    # ... (Your chat logic remains here)
    pass

@app.route("/summarize", methods=["POST"])
def summarize():
    # ... (Your summarize logic remains here)
    pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
