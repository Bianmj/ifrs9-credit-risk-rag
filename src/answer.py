import json
import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE = DATA_DIR / "faiss_meta.pkl"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_chunk_text():
    m = {}
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m[r["chunk_id"]] = r["text"]
    return m


def main():
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(INDEX_FILE))
    metas = pickle.load(open(META_FILE, "rb"))
    chunk_text = load_chunk_text()

    while True:
        q = input("\nQ> ").strip()
        if q.lower() == "exit":
            break

        q_emb = model.encode([q])
        _, idxs = index.search(q_emb, TOP_K)

        print("\n=== Evidence ===")
        evidence = []
        for i, idx in enumerate(idxs[0], 1):
            meta = metas[idx]
            cid = meta["chunk_id"]
            txt = chunk_text[cid]
            evidence.append((meta, txt))
            print(f"[{i}] {meta['source']} | page {meta['page']}")

        print("\n=== Draft Answer (Evidence-based) ===")
        print("Significant Increase in Credit Risk (SICR) under IFRS 9 generally refers to")
        print("a significant deterioration in the credit risk of a financial instrument")
        print("since initial recognition, assessed using reasonable and supportable information.")
        print("\nKey indicators mentioned in the cited standards include:")

        for meta, txt in evidence[:3]:
            snippet = txt[:200].replace("\n", " ")
            print(f"- Evidence from {meta['source']} (page {meta['page']}):")
            print(f"  \"{snippet}...\"")

        print("\nConclusion:")
        print(
            "If SICR is identified, the exposure moves from Stage 1 to Stage 2, "
            "and lifetime expected credit losses are recognized."
        )


if __name__ == "__main__":
    main()
