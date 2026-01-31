import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE = DATA_DIR / "faiss_meta.pkl"

# ---- Retrieval ----
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_DEFAULT = 5

# ---- LLM ----
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2

SYSTEM_PROMPT_FINANCE = """You are an IFRS 9 credit risk assistant for banks.

Strict compliance rules:
- Use ONLY the provided evidence. Do not use external knowledge.
- If evidence is insufficient, say: "Insufficient evidence in the provided documents."
- Every key claim must have citations like [source p.X].
- Do NOT invent paragraph numbers or sources. Cite ONLY the provided [source p.X].

Answer format (follow exactly):
1) Direct definition (1-2 sentences)
2) Practical assessment criteria (bullets)
3) Stage implication (Stage 1 vs Stage 2 vs Stage 3) — only if supported by evidence
4) What evidence is missing (if any)
"""

def _load_chunk_text_map() -> Dict[str, str]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_FILE}. Run chunk.py first.")
    m = {}
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m[r["chunk_id"]] = r["text"]
    return m


def load_assets() -> Tuple[SentenceTransformer, faiss.Index, List[dict], Dict[str, str]]:
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Missing FAISS index/meta. Run index.py first.")

    emb_model = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(str(INDEX_FILE))
    metas = pickle.load(open(META_FILE, "rb"))
    chunk_text_map = _load_chunk_text_map()
    return emb_model, index, metas, chunk_text_map


def retrieve(query: str, emb_model, index, metas, chunk_text_map, top_k: int = TOP_K_DEFAULT) -> List[dict]:
    q_emb = emb_model.encode([query])
    _, idxs = index.search(q_emb, top_k)
    evs = []
    for idx in idxs[0]:
        meta = metas[idx]
        cid = meta["chunk_id"]
        evs.append({
            "source": meta["source"],
            "page": meta["page"],
            "chunk_id": cid,
            "text": chunk_text_map.get(cid, "")
        })
    return evs


def build_evidence_block(evs: List[dict]) -> str:
    blocks = []
    for i, ev in enumerate(evs, 1):
        t = (ev["text"] or "").strip()
        if len(t) > 1400:
            t = t[:1400] + " ..."
        blocks.append(f"Evidence {i}: [{ev['source']} p.{ev['page']}]\n{t}")
    return "\n\n".join(blocks)


def generate_with_openai(query: str, evidence_block: str) -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env at project root.")

    client = OpenAI()

    user_prompt = f"""Question:
{query}

Evidence:
{evidence_block}

Task:
Answer ONLY using the evidence. Add citations [source p.X] for key claims.
If evidence is insufficient, say so and list what is missing.
"""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_FINANCE},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
    )
    return resp.choices[0].message.content
