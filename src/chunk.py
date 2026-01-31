import json
from pathlib import Path
from typing import List

# === 项目根目录（不管从哪里运行都不会错）===
BASE_DIR = Path(__file__).resolve().parent.parent
print("BASE_DIR =", BASE_DIR)

IN_FILE = BASE_DIR / "data" / "processed" / "pages.jsonl"
OUT_FILE = BASE_DIR / "data" / "processed" / "chunks.jsonl"

CHUNK_SIZE = 1200
OVERLAP = 200


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be > overlap")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def main():
    print("Looking for pages file:", IN_FILE)
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing {IN_FILE}. Run src/ingest.py first.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with IN_FILE.open("r", encoding="utf-8") as fin, OUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            text = rec.get("text", "")
            source = rec.get("source", "unknown")
            page = rec.get("page", -1)

            chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
            for j, ch in enumerate(chunks):
                out = {
                    "chunk_id": f"{source}::p{page}::c{j}",
                    "source": source,
                    "page": page,
                    "text": ch
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"✅ Done. Wrote {total_chunks} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()

