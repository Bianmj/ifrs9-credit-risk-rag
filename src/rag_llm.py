import os
import json
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

# ===== Paths =====
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE = DATA_DIR / "faiss_meta.pkl"

# ===== Retrieval settings =====
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# ===== LLM settings =====
LLM_MODEL = "gpt-4o-mini"   # 便宜且很适合RAG
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are a credit risk / IFRS 9 assistant.
Rules:
1) Use ONLY the provided evidence to answer.
2) If evidence is insufficient, say so and ask for more documents.
3) Cite sources for key claims using format: [source p.X]
4) Be concise and structured.
"""


def load_chunk_text_map() -> Dict[str, str]:
    m = {}
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m[r["chunk_id"]] = r["text"]
    return m


def retrieve(query: str, index, emb_model, metas, chunk_text_map, top_k: int = TOP_K) -> List[Dict]:
    q_emb = emb_model.encode([query])
    _, indices = index.search(q_emb, top_k)

    evs = []
    for idx in indices[0]:
        meta = metas[idx]
        cid = meta["chunk_id"]
        evs.append({
            "source": meta["source"],
            "page": meta["page"],
            "chunk_id": cid,
            "text": chunk_text_map.get(cid, "")
        })
    return evs


def build_evidence_block(evs: List[Dict]) -> str:
    blocks = []
    for i, ev in enumerate(evs, 1):
        text = ev["text"].strip()
        # 控制长度，避免prompt太大
        if len(text) > 1400:
            text = text[:1400] + " ..."
        blocks.append(f"Evidence {i}: [{ev['source']} p.{ev['page']}]\n{text}")
    return "\n\n".join(blocks)


def main():
    # Load API key
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env at project root.")

    client = OpenAI()

    # Check files
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Missing FAISS index/meta. Run: python src/index.py")
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("Missing chunks.jsonl. Run: python src/chunk.py")

    print("Loading embedding model:", EMBED_MODEL)
    emb_model = SentenceTransformer(EMBED_MODEL)

    print("Loading FAISS index:", INDEX_FILE)
    index = faiss.read_index(str(INDEX_FILE))
    metas = pickle.load(open(META_FILE, "rb"))

    print("Loading chunk texts...")
    chunk_text_map = load_chunk_text_map()

    print("\n✅ RAG+LLM ready. Type a question (or 'exit').")
    while True:
        query = input("\nQ> ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        evs = retrieve(query, index, emb_model, metas, chunk_text_map, TOP_K)
        evidence_block = build_evidence_block(evs)

        user_prompt = f"""Question:
{query}

Evidence:
{evidence_block}

Task:
Answer ONLY using the evidence. Add citations like [source p.X] for key claims.
"""

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
        )

        answer = resp.choices[0].message.content

        print("\n=== Answer ===")
        print(answer)

        print("\n=== Retrieved sources ===")
        for ev in evs:
            print(f"- {ev['source']} p.{ev['page']} ({ev['chunk_id']})")


if __name__ == "__main__":
    main()
