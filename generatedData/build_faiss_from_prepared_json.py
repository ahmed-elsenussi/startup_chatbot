import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch

def build_faiss_from_prepared_json(input_file, faiss_index_path, metadata_output_file):
    input_file = Path(input_file)
    faiss_index_path = Path(faiss_index_path)
    metadata_output_file = Path(metadata_output_file)

    # Make sure output directories exist
    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # Load prepared JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    embeddings = []
    metadata = []

    print(f"Encoding {len(data)} prepared texts...")
    for item in data:
        text = item.get('prepared_text', '')
        emb = model.encode(text, convert_to_numpy=True).astype('float32')
        embeddings.append(emb)

        metadata.append({
            "id": item.get('id'),
            "name": item.get('name'),
            "fieldId": item.get('fieldId'),
            "description": item.get('description'),
            "websiteUrl": item.get('websiteUrl'),
            "email": item.get('email'),
            "phone": item.get('phone'),
            "facebookUrl": item.get('facebookUrl'),
            "address": item.get('address'),
            "logoImageUrl": item.get('logoImageUrl'),
            "prepared_text": text
        })

    embeddings_np = np.array(embeddings, dtype='float32')
    dimension = embeddings_np.shape[1]

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    faiss.write_index(index, str(faiss_index_path))
    print(f"FAISS index saved to {faiss_index_path}")

    # Save metadata
    with open(metadata_output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to {metadata_output_file}")

# Example usage
input_file = "prepared_data.json"
faiss_index_path = "data/new_company_chunks.index"
metadata_output_file = "data/new_company_chunks_metadata.json"

build_faiss_from_prepared_json(input_file, faiss_index_path, metadata_output_file)
