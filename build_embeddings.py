"""Rebuild and persist the healthcare RAG FAISS index from knowledge_base/*.txt."""
from dotenv import load_dotenv
load_dotenv()

from rag_engine import RAGEngine, INDEX_PATH, CHUNKS_PATH

# Always rebuild so newly added/edited .txt files are included.
INDEX_PATH.unlink(missing_ok=True)
CHUNKS_PATH.unlink(missing_ok=True)

engine = RAGEngine()
if not engine.ready:
    raise SystemExit(f"Embedding/index build failed: {engine.error}")

print(f"Healthcare RAG index is ready: {len(engine.chunks)} chunks indexed.")
print("Saved files:")
print(" - storage/medical.faiss")
print(" - storage/chunks.pkl")
