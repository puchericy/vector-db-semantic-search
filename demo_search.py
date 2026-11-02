"""Прикладні запити до векторної БД (розділ 3): пошук за змістом, фільтри, міжмовний пошук, порівняння з BM25.
Потрібна база chroma_db (створюється build_chroma.py). Запуск: py demo_search.py
Результат: demo_report.txt
"""
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"
TOP = 3
client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("kb_chunks")
model = SentenceTransformer(MODEL)

# --- дані для BM25 (той самий корпус, що в колекції) ---
allr = col.get(include=["documents", "metadatas"])
bm = {}
for lang in ("uk", "en"):
    idx = [i for i, m in enumerate(allr["metadatas"]) if m["lang"] == lang]
    docs = [allr["documents"][i] for i in idx]
    bm[lang] = (BM25Okapi([re.findall(r"\w+", d.lower()) for d in docs]), idx)

report = []


def log(s=""):
    print(s)
    report.append(s)


def snippet(t, n=110):
    t = t.replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


def vector_search(query, where, by_article=True):
    qv = model.encode(["query: " + query], normalize_embeddings=True)[0].tolist()
    res = col.query(query_embeddings=[qv], n_results=40, where=where, include=["metadatas", "documents", "distances"])
    out, seen = [], set()
    for m, d, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
        key = m["article_id"] if by_article else (m["article_id"], m["lang"], m["chunk_index"])
        if key in seen:
            continue
        seen.add(key)
        out.append((m["article_id"], m["lang"], m["title"], dist, d))
        if len(out) == TOP:
            break
    return out


def bm25_search(query, lang):
    model_bm25, idx = bm[lang]
    sc = model_bm25.get_scores(re.findall(r"\w+", query.lower()))
    order = [k for k in sorted(range(len(sc)), key=lambda k: -sc[k]) if sc[k] > 0]
    out, seen = [], set()
    for k in order:
        m = allr["metadatas"][idx[k]]
        if m["article_id"] in seen:
            continue
        seen.add(m["article_id"])
        out.append((m["article_id"], m["lang"], m["title"], sc[k], allr["documents"][idx[k]]))
        if len(out) == TOP:
            break
    return out


DEMO = [
    ("Д1", "ФВ-4 пошук за змістом", "Як створити модель даних?", {"lang": "uk"}, None, False),
    ("Д2", "ФВ-4, ФВ-5 фільтр за мовою", "Як забезпечити тестувальників реалістичними тестовими даними, водночас захищаючи конфіденційну інформацію клієнтів?", {"lang": "uk"}, "uk", False),
    ("Д3", "ФВ-6 українське питання, англійські статті", "Як забезпечити тестувальників реалістичними тестовими даними, водночас захищаючи конфіденційну інформацію клієнтів?", {"lang": "en"}, "en", False),
    ("Д4", "ФВ-6 англійське питання, українські статті", "What is the difference between active and passive replication?", {"lang": "uk"}, "uk", False),
    ("Д5", "ФВ-5 фільтр за статтею: найкращі фрагменти всередині статті 41", "How does single-leader database replication work?", {"$and": [{"article_id": 41}, {"lang": "en"}]}, None, True),
    ("Д6", "Пошук без фільтра за мовою", "Чому пошук потрібної інформації серед великої кількості даних займає багато часу та які методи дозволяють оптимізувати цей процес?", None, None, False),
    ("Д7", "Запит поза предметною областю (перевірка обмежень)", "Як приготувати борщ?", {"lang": "uk"}, "uk", False),
]

for did, fv, q, where, bm_lang, by_chunk in DEMO:
    log("=" * 78)
    log(f"{did} | {fv}")
    log(f"Запит: {q}")
    log(f"Фільтр: {where}")
    log("Вектори (відстань = 1 − косинусна схожість; менше = ближче):")
    for a, l, t, dist, d in vector_search(q, where, by_article=not by_chunk):
        log(f"   {a:>3} | {l} | {t[:34]:34s} | {dist:.3f} | {snippet(d)}")
    if bm_lang:
        log(f"BM25 (за ключовими словами, мова {bm_lang}):")
        res = bm25_search(q, bm_lang)
        if not res:
            log("   збігів немає")
        for a, l, t, s, d in res:
            log(f"   {a:>3} | {l} | {t[:34]:34s} | {s:.2f} | {snippet(d)}")

with open("demo_report.txt", "w", encoding="utf-8-sig") as f:
    f.write("\n".join(report))
print("\nЗвіт: demo_report.txt")
