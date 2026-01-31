from pathlib import Path

p = Path(__file__).resolve().parent.parent / "data" / "processed" / "pages.jsonl"

print("Checking path:", p)
print("Exists:", p.exists())

if p.exists():
    print("Size(bytes):", p.stat().st_size)
