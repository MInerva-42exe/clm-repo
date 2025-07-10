import os
import json
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
from itertools import cycle
import google.api_core.exceptions

# --- Setup ---
load_dotenv()
app = Flask(__name__)

# --- Database & AI Key Setup ---
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env file.")
# engine = create_engine(DATABASE_URL) # Your DB engine

GEMINI_API_KEYS_STR = os.environ.get("GEMINI_API_KEYS")
if not GEMINI_API_KEYS_STR:
    raise ValueError("GEMINI_API_KEYS is not set in .env file.")
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_STR.split(',')]
api_key_cycler = cycle(GEMINI_API_KEYS)
try:
    genai.configure(api_key=next(api_key_cycler))
except StopIteration:
    raise ValueError("The API key list is empty.")

def generate_content_with_failover(*args, **kwargs):
    keys_to_try = len(GEMINI_API_KEYS)
    for _ in range(keys_to_try):
        try:
            tools = kwargs.pop('tools', None)
            model = genai.GenerativeModel(model_name='gemini-1.5-flash', tools=tools)
            return model.generate_content(*args, **kwargs)
        except (google.api_core.exceptions.PermissionDenied, google.api_core.exceptions.ResourceExhausted) as e:
            print(f"API key failed: {e}. Switching keys.")
            new_key = next(api_key_cycler)
            genai.configure(api_key=new_key)
            continue
    raise Exception("All API keys failed.")

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
                    'keywords': genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING))
                }
            )
        )
    ]
)

# --- Final System Prompt ---
SYSTEM_PROMPT = f"""
You are WSM Content Assistant, a friendly and highly focused AI expert on software documentation.
Your sole job is to help users find documents. You will either call the `search_database` tool, ask a clarifying question, or handle a simple greeting.

=== 1. INPUT PROCESSING PRIORITY ===
1.  **PRODUCT NORMALIZATION:** Map any user-provided product name or acronym to its official name using this map: {json.dumps(PRODUCT_ACRONYM_MAP)}. This is your highest priority.
2.  **DOCUMENT-TYPE MAPPING:** Map the user's request to one of the following specific categories: {json.dumps(VALID_DOC_TYPES)}.
3.  **KEYWORD EXTRACTION:** Extract 1-3 core concepts from the user's query to use as keywords. Focus on phrases, not single words.

=== 2. DECISION LOGIC ===
-   **If the user's request contains enough information to search (a product, doc type, or keyword):** Your ONLY action is to call the `search_database` tool with the parameters you have derived.
-   **If the user's request is too vague or missing key information:** Ask a clear, concise clarifying question.
-   **If the user's message is a simple greeting or casual chat:** Respond naturally and guide them toward a search.

=== 3. EXAMPLES ===
-   **Greeting:** User: "hi" -> Your Response: "Hello! How can I help you find a document today?"
-   **Acronym Lookup:** User: "show me ADMP docs" -> Your Action: Call `search_database` with `product="ADManager Plus"`.
-   **Document Mapping:** User: "any whitepapers on m365 manager plus?" -> Your Action: Call `search_database` with `product="M365 Manager Plus"` and `document_type="E-book or guide"`.
-   **Vague Request:** User: "I need a comparison sheet" -> Your Response: "Certainly. Which products would you like to compare?"
-   **Complex Query:** User: "Find technical docs about security compliance in ADAudit Plus" -> Your Action: Call `search_database` with `product="ADAudit Plus"`, `document_type="Technical Document"`, `keywords=["security compliance"]`.
-   **Follow‑up:** History shows a search for AD360. User: "okay, now just videos" -> Your Action: Call `search_database` with `product="AD360"` and `document_type="Video"`.
"""

def search_database(product: str = None, document_type: str = None, keywords: list = None):
    # This is a placeholder for your actual database search logic.
    print(f"DATABASE SEARCH: Product={product}, Type={document_type}, Keywords={keywords}")
    return [{"Product": product or "ADManager Plus", "Doc_type": document_type or "Case study", "Content_Title": "Example Document Title", "Description": "This is an example document description from the database.", "Link": "#"}]

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        full_prompt = [SYSTEM_PROMPT] + history + [{'role': 'user', 'parts': [{'text': user_message}]}]

        response = generate_content_with_failover(
            full_prompt,
            tools=[search_tool]
        )
        
        response_part = response.candidates[0].content.parts[0]

        # Use the safer check for the function call
        if response_part and getattr(response_part, 'function_call', None) and response_part.function_call.name == "search_database":
            params = {k: v for k, v in response_part.function_call.args.items()}
            documents = search_database(**params)
            
            if documents:
                return jsonify({"type": "documents", "message": f"I found {len(documents)} document(s) for you:", "data": documents})
            else:
                return jsonify({"type": "conversation", "message": "I couldn't find any documents that match your request."})
        else:
            # Safely get the text from the response
            response_text = response_part.text if response_part and hasattr(response_part, 'text') else response.text
            return jsonify({"type": "conversation", "message": response_text})

    except Exception as e:
        print(f"An error occurred in the chat endpoint: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

if __name__ == '__main__':
    app.run(debug=True)
