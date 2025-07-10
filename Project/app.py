import os
import json
from itertools import cycle

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import google.api_core.exceptions

# --- Setup ---
load_dotenv()
app = Flask(__name__)

# --- Database & AI Key Setup ---
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env file.")

# Use the safer method for loading keys
GEMINI_API_KEYS_STR = os.environ.get("GEMINI_API_KEYS")
if not GEMINI_API_KEYS_STR:
    raise ValueError("GEMINI_API_KEYS is not set in .env file.")
GEMINI_API_KEYS = [k.strip() for k in GEMINI_API_KEYS_STR.split(',') if k.strip()]
if not GEMINI_API_KEYS:
    raise ValueError("No valid Gemini API keys found in GEMINI_API_KEYS.")
api_key_cycler = cycle(GEMINI_API_KEYS)
genai.configure(api_key=next(api_key_cycler))


def generate_content_with_failover(*args, **kwargs):
    attempts = len(GEMINI_API_KEYS)
    for _ in range(attempts):
        try:
            model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                tools=kwargs.pop('tools', None),
                system_instruction=kwargs.pop('system_instruction', None)
            )
            return model.generate_content(*args, **kwargs)
        except (google.api_core.exceptions.PermissionDenied,
                google.api_core.exceptions.ResourceExhausted) as e:
            print(f"Key failed: {e}. Rotating key.")
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

# Using the superior, structured prompt
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

=== 3. EXAMPLES ===
1. **Greeting**
   User: “hi”
   → “Hello! Which product or document type are you interested in?”

2. **Acronym Lookup**
   User: “show me ADMP docs”
   → tool call with `product="ADManager Plus"`

3. **Whitepaper Request**
   User: “any whitepapers on m365 manager plus?”
   → tool call with `document_type="E-book or guide"`

4. **Vague Guide Request**
   User: “I need a guide.”
   → “Sure—what product is the guide for?”

5. **Complex Query**
   User: “Find technical docs about security compliance in ADAudit Plus”
   → tool call with `product="ADAudit Plus"`, `document_type="Technical Document"`, `keywords=["security compliance"]`

6. **Follow-up**
   History: previous search for AD360
   User: “Now just the case studies.”
   → tool call with `product="AD360"`, `document_type="Case study"`
"""


def search_database(product: str = None, document_type: str = None, keywords: list = None):
    print(f"DATABASE SEARCH: Product={product}, Type={document_type}, Keywords={keywords}")
    # Placeholder for your actual database query logic
    return [{
        "Product": product or "ADManager Plus",
        "Doc_type": document_type or "Case study",
        "Content_Title": "Example Document Title",
        "Description": "This is an example document description from the database.",
        "Link": "#"
    }]


# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_message = data.get('message', '').strip()
    history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        conversation_history = history + [{'role': 'user', 'parts': [{'text': user_message}]}]

        response = generate_content_with_failover(
            conversation_history,
            tools=[search_tool],
            system_instruction=SYSTEM_PROMPT
        )
        
        response_part = response.candidates[0].content.parts[0]

        # Use the correct, functional logic for handling the tool call
        if response_part and getattr(response_part, 'function_call', None) and response_part.function_call.name == "search_database":
            params = {k: v for k, v in response_part.function_call.args.items()}
            documents = search_database(**params)
            
            if documents:
                return jsonify({
                    "type": "documents",
                    "message": f"I found {len(documents)} document(s) for you:",
                    "data": documents
                })
            else:
                return jsonify({
                    "type": "conversation",
                    "message": "I couldn't find any documents that match your request."
                })
        else:
            # Safely get the text from the response for conversational replies
            response_text = response_part.text if response_part and hasattr(response_part, 'text') else response.text
            return jsonify({"type": "conversation", "message": response_text})

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500


if __name__ == '__main__':
    app.run(debug=True)
