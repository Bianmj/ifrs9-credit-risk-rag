import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# === 项目根目录 ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE = DATA_DIR / "faiss_meta.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_FILE}, run chunk.py first")

    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    texts = []
    metas = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            metas.append({
                "source": rec["source"],
                "page": rec["page"],
                "chunk_id": rec["chunk_id"]
            })

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    with META_FILE.open("wb") as f:
        pickle.dump(metas, f)

    print("✅ FAISS index built")
    print("Index file:", INDEX_FILE)
    print("Meta file:", META_FILE)


if __name__ == "__main__":
    main()
