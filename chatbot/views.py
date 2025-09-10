import json
from pathlib import Path
import faiss
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import torch
from django.conf import settings
import re

# --------------------------
# Paths to FAISS index + metadata
# --------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = BASE_DIR / "generatedData/data/new_company_chunks.index"
METADATA_FILE = BASE_DIR / "generatedData/data/new_company_chunks_metadata.json"

# Load FAISS index and metadata once
index = faiss.read_index(str(FAISS_INDEX_PATH))
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)  # secure
gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# --------------------------
# Constants
# --------------------------
UMBRELLA_TYPES = ["Talent", "Networking", "Growth", "Support", "Funding"]
INCLUDE_EMPTY_TYPES = False

NORMAL_RESPONSES = {
    "hi": "Hello! I'm here to help you find companies that can support your startup idea.",
    "hello": "Hi there! Tell me about your project and I can suggest compatible companies.",
    "hey": "Hey! I can help you find companies or institutes suited for your startup idea.",
    "what is this app about": "This app suggests compatible companies or institutes based on your startup idea description.",
    "what's up": "I'm here to help you find companies that match your startup needs.",
    "sup": "Hello! I can suggest companies that might support your project.",
    "thanks": "You're welcome! Let me know if you want suggestions for companies.",
    "thank you": "You're welcome! I can help you find compatible companies for your startup.",
}

# --------------------------
# Helper: embed + normalize
# --------------------------
def embed_and_normalize(text: str) -> np.ndarray:
    emb = embed_model.encode([text], convert_to_numpy=True).astype("float32")
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)

# --------------------------
# API Endpoint
# --------------------------
@csrf_exempt
def suggest_companies(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required."}, status=400)

    try:
        data = json.loads(request.body)
        user_prompt_raw = data.get("prompt", "").strip()
        if not user_prompt_raw:
            return JsonResponse({"error": "Prompt is required."}, status=400)

        user_prompt_lower = user_prompt_raw.lower()

        # Greeting / exact casual match only
        for key, reply in NORMAL_RESPONSES.items():
            if user_prompt_lower == key:
                return JsonResponse({"message": reply, "types": []})

        # ---------------------------
        # Always query Gemini, let it decide
        # ---------------------------
        # Embed & search FAISS
        user_emb = embed_and_normalize(user_prompt_raw)
        k = 10
        D, I = index.search(user_emb.astype("float32"), k)
        context_chunks = [metadata[i]["prepared_text"] for i in I[0]]
        context_text = "\n".join(context_chunks)

        # Prompt Gemini
        final_prompt = (
            "You are an assistant for suggesting companies based on a user's startup idea.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User question: {user_prompt_raw}\n\n"
            "Task:\n"
            "Decide what to return:\n"
            "- If the user is just greeting, casual chatting, or not asking about companies/startups, "
            "return JSON in this format:\n"
            "{ \"message\": \"<short helpful message>\", \"types\": [] }\n\n"
            "- If the user is asking about companies/startups, return structured JSON:\n"
            "{\n"
            "  \"message\": \"<short summary>\",\n"
            "  \"types\": [\n"
            "    {\n"
            "      \"type\": \"<one of: Talent, Networking, Growth, Support, Funding>\",\n"
            "      \"companies\": [\n"
            "        {\n"
            "          \"name\": \"<company name>\",\n"
            "          \"reason\": \"<why relevant>\",\n"
            "          \"fields\": []\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "⚠️ Rules:\n"
            "- Always return valid JSON only.\n"
            "- Do not include anything besides JSON.\n"
        )

        response = gemini_model.generate_content(final_prompt)

        # Try to parse JSON
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(json)?", "", raw_text)
            raw_text = raw_text.rstrip("`").strip()

        try:
            structured_data = json.loads(raw_text)
        except Exception:
            structured_data = {"message": "AI failed to return valid JSON", "types": []}

        # ---------------------------
        # Enrich metadata
        # ---------------------------
        types = structured_data.get("types", [])
        for t in types:
            for company in t.get("companies", []):
                meta = next((m for m in metadata if m["name"] == company["name"]), None)
                if meta:
                    field_val = meta.get("fieldId", [])
                    if not isinstance(field_val, list):
                        field_val = [field_val]
                    company["fields"] = field_val
                    for key, value in meta.items():
                        if key not in ["prepared_text", "fieldId", "name"] and value is not None:
                            company[key] = value
                    if company.get("logoImageUrl"):
                        filename = Path(company["logoImageUrl"]).name
                        company["logoImageUrl"] = request.build_absolute_uri(
                            f"{settings.STATIC_URL}images/{filename}"
                        )

        # Remove empty types if configured
        if not INCLUDE_EMPTY_TYPES:
            types = [t for t in types if t.get("companies")]

        structured_data["types"] = types
        return JsonResponse(structured_data)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
