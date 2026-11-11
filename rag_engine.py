import os
import pickle
from pathlib import Path

import numpy as np
import faiss
from google import genai
from groq import Groq

BASE = Path(__file__).resolve().parent
KB = BASE / "knowledge_base"
CACHE = BASE / "storage"
INDEX_PATH = CACHE / "medical.faiss"
CHUNKS_PATH = CACHE / "chunks.pkl"

class RAGEngine:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.ready = False
        self.error = None
        CACHE.mkdir(exist_ok=True)
        try:
            self._load_or_build()
            self.ready = True
        except Exception as exc:
            self.error = str(exc)

    def _client(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY to enable document retrieval.")
        return genai.Client(api_key=key)

    def _read_documents(self):
        docs = []
        for path in KB.glob("*.txt"):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append((path.name, text))
        if not docs:
            raise RuntimeError("No .txt reference documents found in knowledge_base/.")
        return docs

    def _chunk(self, docs, size=850, overlap=140):
        chunks = []
        for source, text in docs:
            start = 0
            while start < len(text):
                part = text[start:start + size].strip()
                if part:
                    chunks.append({"source": source, "text": part})
                start += size - overlap
        return chunks

    def _embed(self, client, text):
        response = client.models.embed_content(
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
            contents=text
        )
        return response.embeddings[0].values

    def _load_or_build(self):
        if INDEX_PATH.exists() and CHUNKS_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            with CHUNKS_PATH.open("rb") as f:
                self.chunks = pickle.load(f)
            return
        client = self._client()
        self.chunks = self._chunk(self._read_documents())
        vectors = [self._embed(client, item["text"]) for item in self.chunks]
        matrix = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        faiss.write_index(self.index, str(INDEX_PATH))
        with CHUNKS_PATH.open("wb") as f:
            pickle.dump(self.chunks, f)

    def retrieve(self, question, top_k=4):
        client = self._client()
        vector = np.asarray([self._embed(client, question)], dtype=np.float32)
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, top_k)
        found = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                item = self.chunks[int(idx)]
                found.append({
                    "source": item["source"],
                    "text": item["text"],
                    "similarity": round(float(score), 4)
                })
        return found

    def ask(self, question):
        if not self.ready:
            raise RuntimeError(self.error or "RAG engine is not ready.")
        results = self.retrieve(question)
        if not results:
            return {"answer": "No relevant reference passages were found.", "mode": "knowledge_retrieval", "sources": []}
        context = "\n\n".join(
            f"[Source: {item['source']} | similarity: {item['similarity']}]\n{item['text']}"
            for item in results
        )
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Set GROQ_API_KEY to generate answers from retrieved documents.")
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=[
                {"role": "system", "content": (
                    "Answer using only the supplied reference context. "
                    "If the context does not support an answer, say so. "
                    "This is educational information, not diagnosis or treatment advice. "
                    "Cite source filenames in square brackets."
                )},
                {"role": "user", "content": f"Reference context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.1
        )
        answer = response.choices[0].message.content.strip()
        sources = [{"source": x["source"], "similarity": x["similarity"]} for x in results]
        return {"answer": answer, "mode": "rag_knowledge", "sources": sources, "retrieved_chunks": results}
