import json
import numpy as np
import faiss
from django.core.management.base import BaseCommand
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch

class Command(BaseCommand):
    help = "Generate embeddings from text chunks and build FAISS vector index"

    def add_arguments(self, parser):
        parser.add_argument('input_file', type=str, help='Input JSON file with text chunks')
        parser.add_argument('--faiss_index_path', type=str, default='data/company_chunks.index',
                            help='Output path for FAISS index (.index)')
        parser.add_argument('--metadata_output_file', type=str, default='data/company_chunks_metadata.json',
                            help='Output path for metadata JSON')

    def handle(self, *args, **kwargs):
        input_file = Path(kwargs['input_file'])
        faiss_index_path = Path(kwargs['faiss_index_path'])
        metadata_output_file = Path(kwargs['metadata_output_file'])

        # Make sure output directories exist
        faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_output_file.parent.mkdir(parents=True, exist_ok=True)

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stdout.write(f"Using device: {device}")

        # Load SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        # Load text chunks
        with open(input_file, 'r', encoding='utf-8') as f:
            text_chunks = json.load(f)

        embeddings = []
        metadata = []

        self.stdout.write(f"Encoding {len(text_chunks)} text chunks...")
        for chunk in text_chunks:
            emb = model.encode(chunk['text'], convert_to_numpy=True).astype('float32')
            embeddings.append(emb)
            metadata.append({
                "id": chunk.get('id'),
                "text": chunk.get('text'),
                "logo_image": chunk.get('logo_image', None),
            })

        embeddings_np = np.array(embeddings, dtype='float32')
        dimension = embeddings_np.shape[1]

        self.stdout.write("Building FAISS index...")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
        faiss.write_index(index, str(faiss_index_path))
        self.stdout.write(self.style.SUCCESS(f"FAISS index saved to {faiss_index_path}"))

        # Save metadata
        with open(metadata_output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        self.stdout.write(self.style.SUCCESS(f"Metadata saved to {metadata_output_file}"))
