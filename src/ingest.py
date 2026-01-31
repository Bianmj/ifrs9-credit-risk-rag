from pathlib import Path
import json
from tqdm import tqdm
from pypdf import PdfReader

# === 项目根目录 ===
BASE_DIR = Path(__file__).resolve().parent.parent
print("BASE_DIR =", BASE_DIR)

RAW_DIR = BASE_DIR / "data" / "raw_pdfs"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_FILE = OUT_DIR / "pages.jsonl"


def extract_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        yield {
            "source": pdf_path.name,
            "page": i + 1,
            "text": text
        }


def main():
    print("Looking for PDFs in:", RAW_DIR)

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Directory does not exist: {RAW_DIR}")

    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {RAW_DIR.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for pdf in tqdm(pdf_files, desc="Ingest PDFs"):
            for rec in extract_pages(pdf):
                if len(rec["text"]) < 30:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

    print(f"✅ Done. Wrote {count} pages to {OUT_FILE}")


if __name__ == "__main__":
    main()
