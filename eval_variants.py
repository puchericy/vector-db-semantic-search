"""Порівняння варіантів пошуку на однакових 72 парних статтях.

Варіанти: V1 базовий; V2 + назва статті у фрагменті; V3 без розділів «Примітки/Посилання»; V4 = V2 + V3.
Методи:   vector, bm25, hybrid (злиття двох списків за рангами, RRF).
Запуск:   py eval_variants.py            (запити з queries.json)
          py eval_variants.py file.json  (інший набір запитів, наприклад контрольний)
"""
import csv
import json
import os
import random
import re
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"
QFILE = sys.argv[1] if len(sys.argv) > 1 else "queries.json"
MAX_LEN, MIN_LEN, RRF_K = 800, 100, 60

STOP_HEADERS = {"примітки", "посилання", "джерела", "література", "див. також", "зовнішні посилання",
                "references", "external links", "further reading", "see also", "notes",
                "bibliography", "sources", "citations", "footnotes"}

with open("articles_uk_en.json", encoding="utf-8") as f:
    records = json.load(f)
with open("chunks.json", encoding="utf-8") as f:
    old_chunks = json.load(f)
with open(QFILE, encoding="utf-8") as f:
    queries = json.load(f)
old_emb = np.load("embeddings.npy")
model = SentenceTransformer(MODEL)

paired = [i for i, r in enumerate(records) if r["title_en"] and r["text_uk"] and r["text_en"]]
print("Парних статей:", len(paired), "| запитів:", len(queries), "| файл:", QFILE)


# ---------- побудова фрагментів ----------
def split_text(text):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip() and not p.strip().startswith("==")]
    chunks, current = [], ""
    for p in paragraphs:
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


def clean_text(t):
    out = []
    for line in t.split("\n"):
        m = re.match(r"^=+\s*(.*?)\s*=+\s*$", line.strip())
        if m:
            if m.group(1).lower() in STOP_HEADERS:
                break          # далі йдуть примітки/посилання
            continue
        out.append(line)
    return "\n".join(out)


def title_of(a, lang):
    return records[a]["title_" + lang]


def build(variant):
    """Повертає (список фрагментів, матриця ембедингів)."""
    use_title = variant in ("V2", "V4")
    cleaned = variant in ("V3", "V4")
    if not cleaned:
        pos = [i for i, c in enumerate(old_chunks) if c["has_pair"]]
        chunks = [dict(old_chunks[i]) for i in pos]
        base_emb = old_emb[pos]
    else:
        chunks = []
        for a in paired:
            for lang in ("uk", "en"):
                for i, c in enumerate(split_text(clean_text(records[a]["text_" + lang]))):
                    chunks.append({"article_id": a, "lang": lang, "title": title_of(a, lang),
                                   "chunk_index": i, "text": c})
        base_emb = None
    for c in chunks:
        c["doc"] = (c["title"] + ". " + c["text"]) if use_title else c["text"]
    cache = f"emb_{variant}.npy"
    if variant == "V1":
        emb = base_emb
    elif os.path.exists(cache) and len(np.load(cache)) == len(chunks):
        emb = np.load(cache)
    else:
        t0 = time.time()
        emb = model.encode(["passage: " + c["doc"] for c in chunks], batch_size=32,
                           normalize_embeddings=True, show_progress_bar=False)
        np.save(cache, emb)
        print(f"  {variant}: ембединги обчислено за {time.time() - t0:.0f} с")
    return chunks, emb


qvecs = model.encode(["query: " + q["query"] for q in queries], normalize_embeddings=True)


def words(s):
    return re.findall(r"\w+", s.lower())


def first_rank(order, chunks, idxl, targets):
    seen = []
    for k in order:
        a = chunks[idxl[k]]["article_id"]
        if a not in seen:
            seen.append(a)
        if a in targets:
            return len(seen)
    return None


VARIANTS = {"V1": "базовий", "V2": "+ назва у фрагменті", "V3": "без приміток/посилань", "V4": "назва + без приміток"}
METHODS = ("vector", "bm25", "hybrid")
rows, rr = [], {}

for v in VARIANTS:
    chunks, emb = build(v)
    idx = {l: [i for i, c in enumerate(chunks) if c["lang"] == l] for l in ("uk", "en")}
    E = {l: emb[idx[l]] for l in idx}
    bm = {l: BM25Okapi([words(chunks[i]["doc"]) for i in idx[l]]) for l in idx}
    print(f"{v} ({VARIANTS[v]}): фрагментів {len(chunks)} (uk {len(idx['uk'])}, en {len(idx['en'])})")
    for qi, q in enumerate(queries):
        other = "en" if q["lang"] == "uk" else "uk"
        for scen, lang in (("та сама мова", q["lang"]), ("між мовами", other)):
            if scen == "між мовами" and q.get("pair") == "a59":
                continue
            vs = E[lang] @ qvecs[qi]
            v_order = list(np.argsort(-vs, kind="stable"))
            bs = bm[lang].get_scores(words(q["query"]))
            b_order = [k for k in np.argsort(-bs, kind="stable") if bs[k] > 0]
            fused = {}
            for lst in (v_order, b_order):
                for r, k in enumerate(lst, 1):
                    fused[k] = fused.get(k, 0) + 1 / (RRF_K + r)
            h_order = sorted(fused, key=lambda k: -fused[k])
            for method, order in (("vector", v_order), ("bm25", b_order), ("hybrid", h_order)):
                r = first_rank(order, chunks, idx[lang], set(q["relevant"]))
                rows.append({"variant": v, "id": q["id"], "scenario": scen, "method": method, "rank": r})
                rr.setdefault((v, method, scen), {})[q["id"]] = (1 / r) if r else 0.0


# ---------- підсумки ----------
def summarize(v, method, scen):
    ranks = [r["rank"] for r in rows if r["variant"] == v and r["method"] == method and r["scenario"] == scen]
    n = len(ranks)
    rk = lambda k: sum(1 for x in ranks if x is not None and x <= k) / n
    return n, rk(1), rk(5), rk(10), sum(1 / x for x in ranks if x) / n


def delta_ci(v, method, scen):
    base, cur = rr[("V1", method, scen)], rr[(v, method, scen)]
    d = [cur[i] - base[i] for i in base if i in cur]
    random.seed(1)
    n = len(d)
    boots = sorted(sum(random.choice(d) for _ in range(n)) / n for _ in range(2000))
    return sum(d) / n, boots[50], boots[1949]


for scen in ("та сама мова", "між мовами"):
    print(f"\n=== {scen} (запити з {QFILE}) ===")
    print(f"{'варіант':8s}{'метод':8s}{'n':>4s}{'R@1':>7s}{'R@5':>7s}{'R@10':>7s}{'MRR':>8s}   ΔMRR проти V1 [95% інтервал]")
    for v in VARIANTS:
        for m in METHODS:
            n, r1, r5, r10, mrr = summarize(v, m, scen)
            extra = ""
            if v != "V1":
                d, lo, hi = delta_ci(v, m, scen)
                extra = f"   {d:+.3f} [{lo:+.3f}; {hi:+.3f}]"
            print(f"{v:8s}{m:8s}{n:4d}{r1:7.2f}{r5:7.2f}{r10:7.2f}{mrr:8.3f}{extra}")

out = "results_variants_" + os.path.splitext(os.path.basename(QFILE))[0] + ".csv"
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nДетальні результати:", out)
