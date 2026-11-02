import json
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"
THRESHOLD = 3   # показуємо запити, де правильна стаття нижче за цей ранг

with open("articles_uk_en.json", encoding="utf-8") as f:
    records = json.load(f)
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
with open("queries.json", encoding="utf-8") as f:
    queries = json.load(f)
emb = np.load("embeddings.npy")
model = SentenceTransformer(MODEL)


def words(s):
    return re.findall(r"\w+", s.lower())


idx = {l: [i for i, c in enumerate(chunks) if c["lang"] == l] for l in ("uk", "en")}
E = {l: emb[idx[l]] for l in idx}
bm25 = {l: BM25Okapi([words(chunks[i]["text"]) for i in idx[l]]) for l in idx}
qvecs = model.encode(["query: " + q["query"] for q in queries], normalize_embeddings=True)


def top_articles(order, lang, scores, n=3):
    """Повертає перші n різних статей: (article_id, назва, оцінка, початок кращого фрагмента)."""
    seen, out = [], []
    for k in order:
        c = chunks[idx[lang][k]]
        a = c["article_id"]
        if a in seen:
            continue
        seen.append(a)
        out.append((a, c["title"], float(scores[k]), c["text"][:160].replace("\n", " ")))
        if len(out) == n:
            break
    return out


def rank_of(order, lang, targets):
    seen = []
    for k in order:
        a = chunks[idx[lang][k]]["article_id"]
        if a not in seen:
            seen.append(a)
        if a in targets:
            return len(seen)
    return None


lines = []
for qi, q in enumerate(queries):
    lang = q["lang"]
    vs = E[lang] @ qvecs[qi]
    v_order = np.argsort(-vs, kind="stable")
    bs = bm25[lang].get_scores(words(q["query"]))
    b_order = [k for k in np.argsort(-bs, kind="stable") if bs[k] > 0]
    r = rank_of(v_order, lang, set(q["relevant"]))
    if r is not None and r <= THRESHOLD:
        continue
    rb = rank_of(b_order, lang, set(q["relevant"]))
    lines.append("=" * 78)
    lines.append(f'{q["id"]} [{lang}] {q["query"]}')
    lines.append(f'Очікувано: {q["relevant"]} ({records[q["relevant"][0]]["title_uk"]}) | ранг вектори: {r} | ранг BM25: {rb}')
    lines.append("Вектори, перші 3 статті:")
    for a, t, s, snip in top_articles(v_order, lang, vs):
        lines.append(f"   {a} | {t} | {s:.3f} | {snip}")
    lines.append("BM25, перші 3 статті:")
    for a, t, s, snip in top_articles(b_order, lang, bs):
        lines.append(f"   {a} | {t} | {s:.2f} | {snip}")

with open("error_analysis.txt", "w", encoding="utf-8-sig") as f:
    f.write("\n".join(lines))
print("Запитів із помилками (ранг >", THRESHOLD, "):", sum(1 for l in lines if l.startswith("Очікувано")))
print("Результат у файлі error_analysis.txt")
