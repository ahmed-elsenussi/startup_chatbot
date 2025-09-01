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
FAISS_INDEX_PATH = BASE_DIR / "data/company_chunks.index"
METADATA_FILE = BASE_DIR / "data/company_chunks_metadata.json"

# Load FAISS index and metadata once
index = faiss.read_index(str(FAISS_INDEX_PATH))
with open(METADATA_FILE, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Configure Gemini API
genai.configure(api_key="AIzaSyCQKAOQzagjstrrP9oUsilYU4KEVZnwM2c")  # replace with your API key
gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

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
        query_embedding = embed_model.encode(user_prompt, convert_to_numpy=True).astype("float32")

        # Search FAISS for top-k similar chunks
        k = 5
        D, I = index.search(np.array([query_embedding]), k)
        context_chunks = [metadata[i]["text"] for i in I[0]]

        # Build context string
        context_text = "\n".join(context_chunks)
        final_prompt = (
            "It is an application for suggesting companies based on a user's startup idea. "
            "We are specified for this. Do not answer about unrelated things. Answer using the context below:\n\n"
            f"Context:\n{context_text}\n\n"
            f"User question: {user_prompt}"
        )

        # Call Gemini API
        response = gemini_model.generate_content(final_prompt)

        # Access generated text correctly
        response_text = response.text.strip() if response.text else "No response from AI."

        return JsonResponse({"response": response_text})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
