import csv
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"

with open("articles_uk_en.json", encoding="utf-8") as f:
    records = json.load(f)
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
with open("queries.json", encoding="utf-8") as f:
    queries = json.load(f)
emb = np.load("embeddings.npy")
model = SentenceTransformer(MODEL)

# ---------- допоміжні функції ----------
STOP = set("що як для від між або чи та які який яка яке його цієї цих усього всі всіх "
           "where what how the and for with that this which are can its used use any such "
           "from into than rather over also between within".split())


def words(s):
    return re.findall(r"\w+", s.lower())


def stems(s):
    return {w[:5] for w in words(s) if len(w) >= 4 and w not in STOP}


# індекси за мовами
idx = {l: [i for i, c in enumerate(chunks) if c["lang"] == l] for l in ("uk", "en")}
E = {l: emb[idx[l]] for l in idx}
bm25 = {l: BM25Okapi([words(chunks[i]["text"]) for i in idx[l]]) for l in idx}

qvecs = model.encode(["query: " + q["query"] for q in queries], normalize_embeddings=True)


def search(method, qi, lang):
    if method == "vector":
        return np.argsort(-(E[lang] @ qvecs[qi]), kind="stable")
    sc = bm25[lang].get_scores(words(queries[qi]["query"]))
    order = np.argsort(-sc, kind="stable")
    return [k for k in order if sc[k] > 0]  # фрагменти без жодного збігу слів не вважаємо знайденими


def first_rank(order, lang, targets):
    seen = []
    for k in order:
        a = chunks[idx[lang][k]]["article_id"]
        if a not in seen:
            seen.append(a)
        if a in targets:
            return len(seen)
    return None


def bucket(q):
    qs = stems(q["query"])
    best = 0.0
    for a in q["relevant"]:
        ts = stems(records[a]["text_" + q["lang"]])
        best = max(best, len(qs & ts) / max(len(qs), 1))
    return best, ("легкий" if best >= 0.75 else "середній" if best >= 0.5 else "складний")


# ---------- обчислення ----------
rows = []
for qi, q in enumerate(queries):
    ov, bk = bucket(q)
    other = "en" if q["lang"] == "uk" else "uk"
    for scen, lang in (("та сама мова", q["lang"]), ("між мовами", other)):
        if scen == "між мовами" and q.get("pair") == "a59":
            continue
        for method in ("vector", "bm25"):
            order = search(method, qi, lang)
            for sname, targets in (("основні", set(q["relevant"])),
                                   ("основні+часткові", set(q["relevant"]) | set(q["partial"]))):
                rows.append({"id": q["id"], "query_lang": q["lang"], "scenario": scen,
                             "method": method, "strictness": sname,
                             "overlap": round(ov, 2), "bucket": bk,
                             "rank": first_rank(order, lang, targets)})


def summarize(sel):
    n = len(sel)
    ranks = [r["rank"] for r in sel]
    rk = lambda k: sum(1 for x in ranks if x is not None and x <= k) / n
    mrr = sum(1 / x for x in ranks if x is not None) / n
    return n, rk(1), rk(5), rk(10), mrr


def show(title, keyf, rowfilter):
    print("\n" + title)
    groups = {}
    for r in rows:
        if rowfilter(r):
            groups.setdefault(keyf(r), []).append(r)
    for key in sorted(groups):
        n, r1, r5, r10, mrr = summarize(groups[key])
        print(f"{' | '.join(key):55s} n={n:3d}  R@1={r1:.2f} R@5={r5:.2f} R@10={r10:.2f} MRR={mrr:.3f}")


print("Запитів:", len(queries))
show("=== Загальні результати ===",
     lambda r: (r["scenario"], r["strictness"], r["method"]), lambda r: True)
show("=== За складністю (лише основні відповіді) ===",
     lambda r: (r["scenario"], r["bucket"], r["method"]), lambda r: r["strictness"] == "основні")

with open("results_per_query.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nДетальні результати записано у results_per_query.csv")
