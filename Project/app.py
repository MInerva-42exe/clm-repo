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
   - ⚠️ Important: Do not guess or substitute products that are not listed. 
     For example, do not confuse "AD360" with "Log360", even if they sound similar.
   - Only use exact matches or listed acronyms.
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

# --- Semantic Search ---
@lru_cache(maxsize=128)
def _cached_search(product, document_type, keyword_tuple):
    return tuple(_search_database(product, document_type, list(keyword_tuple)))

def search_database(product=None, document_type=None, keywords=None):
    return list(_cached_search(product or "", document_type or "", tuple(keywords or [])))

def _search_database(product: str = None, document_type: str = None, keywords: list = None):
    app.logger.info(f"VECTOR SEARCH: Product='{product}', Type='{document_type}', Keywords={keywords}")

    query_text = " ".join(keywords or []).strip()
    if not query_text:
        query_text = f"{product or ''} {document_type or ''}".strip()
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

    base_sql = """SELECT id, title, summary, url, product, document_type
                  FROM content
                  ORDER BY 1"""  # Replace with proper vector similarity logic

    with engine.connect() as conn:
        result = conn.execute(text(base_sql)).fetchall()
        return [dict(row._mapping) for row in result]

# --- Routes ---
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "Missing 'message'"}), 400

    chat_session = generate_content_with_failover(
        [user_input],
        tools=[search_tool],
        system_instruction=SYSTEM_PROMPT,
    )

    for response_part in chat_session:
        if response_part.candidates and response_part.candidates[0].content.parts:
            part = response_part.candidates[0].content.parts[0]
            if part.function_call:
                params = {k: v for k, v in part.function_call.args.items()}

                # ✅ Product validation
                product_name = params.get("product")
                if product_name:
                    app.logger.info(f"Gemini interpreted product as: {product_name}")
                    if product_name not in PRODUCT_ACRONYM_MAP:
                        app.logger.warning(f"Gemini returned unknown or incorrect product: {product_name}")
                        return jsonify({
                            "type": "conversation",
                            "message": "I'm not sure which product you're referring to. Could you please clarify?"
                        })

                documents = search_database(**params)
                return jsonify({"type": "documents", "results": documents})

    final_text = chat_session.text.strip()
    return jsonify({"type": "conversation", "message": final_text})

@app.route("/summarize", methods=["POST"])
def summarize():
    # Stub
    return jsonify({"summary": "Summary coming soon."})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
