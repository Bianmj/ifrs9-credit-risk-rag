import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE = DATA_DIR / "faiss_meta.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def main():
    model = SentenceTransformer(MODEL_NAME)

    index = faiss.read_index(str(INDEX_FILE))
    with META_FILE.open("rb") as f:
        metas = pickle.load(f)

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        q_emb = model.encode([query])
        distances, indices = index.search(q_emb, TOP_K)

        print("\nTop results:")
        for rank, idx in enumerate(indices[0], 1):
            meta = metas[idx]
            print(
                f"{rank}. {meta['source']} | page {meta['page']} | {meta['chunk_id']}"
            )


if __name__ == "__main__":
    main()
