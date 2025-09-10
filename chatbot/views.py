import json
from pathlib import Path
import faiss
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import torch

# Paths to FAISS index and metadata
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
genai.configure(api_key="AIzaSyCQKAOQzagjstrrP9oUsilYU4KEVZnwM2c")
gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Fixed 5 umbrella types
UMBRELLA_TYPES = ["Talent", "Networking", "Growth", "Support", "Funding"]

# Toggle: include empty types or not
INCLUDE_EMPTY_TYPES = False


@csrf_exempt
def suggest_companies(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required."}, status=400)

    try:
        data = json.loads(request.body)
        user_prompt = data.get("prompt", "").strip()
        if not user_prompt:
            return JsonResponse({"error": "Prompt is required."}, status=400)

        # Embed user prompt
        query_embedding = embed_model.encode(
            user_prompt, convert_to_numpy=True
        ).astype("float32")

        # Search FAISS for top-k similar chunks
        k = 10
        D, I = index.search(np.array([query_embedding]), k)

        # Use 'prepared_text' for context
        context_chunks = [metadata[i]["prepared_text"] for i in I[0]]
        context_text = "\n".join(context_chunks)

        # Build prompt for Gemini (reasoning only, not fields)
        final_prompt = (
            "You are an assistant for suggesting companies based on a user's startup idea.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User question: {user_prompt}\n\n"
            "Task:\n"
            "Return your answer as JSON with this schema:\n"
            "{\n"
            "  'message': '<short summary>',\n"
            "  'types': [\n"
            "    {\n"
            "      'type': '<one of: Talent, Networking, Growth, Support, Funding>',\n"
            "      'companies': [\n"
            "        {\n"
            "          'name': '<company name>',\n"
            "          'reason': '<why relevant>',\n"
            "          'fields': []  # WILL BE OVERRIDDEN\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "⚠️ Rules:\n"
            "- Only provide 'name' and 'reason'. The 'fields' will be added automatically from dataset.\n"
            "- Do not include anything besides JSON."
        )

        # Call Gemini API
        response = gemini_model.generate_content(final_prompt)

        # Parse JSON safely
        try:
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.lower().startswith("json"):
                    raw_text = raw_text[4:].strip()
            structured_data = json.loads(raw_text)
        except Exception:
            structured_data = {
                "message": "AI failed to return valid JSON",
                "raw": response.text,
            }

        # Replace company 'fields' with metadata values
        types = structured_data.get("types", [])
        for t in types:
            for company in t.get("companies", []):
                # Find company in metadata by name
                meta = next((m for m in metadata if m["name"] == company["name"]), None)
                if meta:
                    company["fields"] = meta.get("fieldId", [])

        # Handle umbrella types
        if INCLUDE_EMPTY_TYPES:
            existing_types = {t["type"] for t in types}
            for umbrella in UMBRELLA_TYPES:
                if umbrella not in existing_types:
                    types.append({"type": umbrella, "companies": []})
        else:
            types = [t for t in types if t.get("companies")]

        structured_data["types"] = types

        return JsonResponse(structured_data)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
