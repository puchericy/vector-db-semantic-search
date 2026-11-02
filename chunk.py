import json
import re

MAX_LEN = 800   # максимальна довжина фрагмента, знаків
MIN_LEN = 100   # коротші фрагменти відкидаємо

def split_text(text):
    # прибираємо заголовки виду "== Розділ =="
    paragraphs = [p.strip() for p in text.split("\n")
                  if p.strip() and not p.strip().startswith("==")]
    chunks, current = [], ""
    for p in paragraphs:
        # довгий абзац ділимо за реченнями
        parts = [p] if len(p) <= MAX_LEN else re.split(r"(?<=[.!?])\s+", p)
        for part in parts:
            if current and len(current) + len(part) + 1 > MAX_LEN:
                chunks.append(current)
                current = part
            else:
                current = (current + " " + part).strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= MIN_LEN]

with open("articles_uk_en.json", encoding="utf-8") as f:
    records = json.load(f)

chunks = []
for article_id, r in enumerate(records):
    has_pair = bool(r["title_en"] and r["text_en"] and r["text_uk"])
    for lang, title, text in (("uk", r["title_uk"], r["text_uk"]),
                              ("en", r["title_en"], r["text_en"])):
        if not text:
            continue
        for i, c in enumerate(split_text(text)):
            chunks.append({"article_id": article_id, "lang": lang, "title": title,
                           "chunk_index": i, "has_pair": has_pair, "text": c})

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

for lang in ("uk", "en"):
    sel = [c for c in chunks if c["lang"] == lang]
    avg = sum(len(c["text"]) for c in sel) // max(len(sel), 1)
    print(lang, "фрагментів:", len(sel), ", середня довжина:", avg)
print("Усього фрагментів:", len(chunks))