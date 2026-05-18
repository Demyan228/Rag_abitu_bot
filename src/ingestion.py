import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt


class BM25Ingestor:
    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    def process_documents(self, documents_dir: Path, output_path: Path):
        """Create one global BM25 index for all chunks across all source documents."""
        document_paths = list(documents_dir.glob("*.json"))
        all_text_chunks = []
        chunk_catalog = []

        for document_path in tqdm(document_paths, desc="Preparing chunks for BM25"):
            with open(document_path, 'r', encoding='utf-8') as file:
                document_data = json.load(file)

            document_id = document_data.get("metainfo", {}).get("sha1_name", document_path.stem)
            chunks = document_data.get("content", {}).get("chunks", []) or []

            for chunk in chunks:
                text = chunk.get("text", "")
                if not text.strip():
                    continue
                all_text_chunks.append(text)
                chunk_catalog.append({
                    "document_id": document_id,
                    "page": chunk.get("page"),
                    "chunk_id": chunk.get("id"),
                    "type": chunk.get("type", "content"),
                    "text": text,
                })

        output_path.mkdir(parents=True, exist_ok=True)
        bm25_index = self.create_bm25_index(all_text_chunks)

        with open(output_path / "bm25.pkl", 'wb') as file:
            pickle.dump(bm25_index, file)
        with open(output_path / "bm25_chunks.json", 'w', encoding='utf-8') as file:
            json.dump(chunk_catalog, file, ensure_ascii=False, indent=2)

        print(f"Prepared global BM25 index from {len(chunk_catalog)} chunks across {len(document_paths)} documents")


class VectorDBIngestor:
    def __init__(self):
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        proxy_pass = os.getenv("PROXY_PASSWORD")
        proxy_user = os.getenv("PROXY_USERNAME")
        if not proxy_pass:
            raise RuntimeError("PROXY_PASSWORD missing")
        if not proxy_user:
            raise RuntimeError("PROXY_USERNAME missing")

        os.environ["HTTP_PROXY"] = f"http://{proxy_user}:{proxy_pass}@5.129.219.79:3128"
        os.environ["HTTPS_PROXY"] = f"http://{proxy_user}:{proxy_pass}@5.129.219.79:3128"

        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-3-large") -> List[float]:
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")

        if isinstance(text, list):
            text_batches = [text[i:i + 512] for i in range(0, len(text), 512)]
        else:
            text_batches = [text]

        embeddings = []
        for batch in text_batches:
            response = self.llm.embeddings.create(input=batch, model=model)
            embeddings.extend([embedding.embedding for embedding in response.data])

        return embeddings

    def _create_vector_db(self, embeddings: List[List[float]]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        return index

    def process_documents(self, documents_dir: Path, output_dir: Path):
        """Create one global FAISS index and metadata catalog for all document chunks."""
        document_paths = list(documents_dir.glob("*.json"))
        all_text_chunks = []
        chunk_catalog = []

        for document_path in tqdm(document_paths, desc="Preparing chunks for vector DB"):
            with open(document_path, 'r', encoding='utf-8') as file:
                document_data = json.load(file)

            document_id = document_data.get("metainfo", {}).get("sha1_name", document_path.stem)
            chunks = document_data.get("content", {}).get("chunks", []) or []

            for chunk in chunks:
                text = chunk.get("text", "")
                if not text.strip():
                    continue
                all_text_chunks.append(text)
                chunk_catalog.append({
                    "document_id": document_id,
                    "page": chunk.get("page"),
                    "chunk_id": chunk.get("id"),
                    "type": chunk.get("type", "content"),
                    "text": text,
                })

        output_dir.mkdir(parents=True, exist_ok=True)
        embeddings = self._get_embeddings(all_text_chunks)
        vector_index = self._create_vector_db(embeddings)

        faiss.write_index(vector_index, str(output_dir / "global.faiss"))
        with open(output_dir / "global_chunks.json", 'w', encoding='utf-8') as file:
            json.dump(chunk_catalog, file, ensure_ascii=False, indent=2)

        print(f"Prepared global vector DB from {len(chunk_catalog)} chunks across {len(document_paths)} documents")
