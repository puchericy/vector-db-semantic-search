"""Побудова векторної БД у Chroma, перевірка пошуку й функціональні тести.

Що робить:
 1. Створює колекцію kb_chunks (HNSW, косинусна відстань) із фрагментів 72 парних статей і готових ембедингів.
 2. Порівнює пошук Chroma (HNSW) з точним пошуком NumPy на обох наборах запитів.
 3. Виконує тести фільтрів, цілісності й резервної копії.
Результат: chroma_db/ (база), chroma_report.txt (звіт).
Запуск: py build_chroma.py
"""
import json
import os
import re
import shutil
import time

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"
DB_PATH, BACKUP_PATH, COLLECTION = "chroma_db", "chroma_db_backup", "kb_chunks"
N_RESULTS = 200          # скільки фрагментів просимо в Chroma для рангування статей

report = []


def log(s=""):
    print(s)
    report.append(s)


# ---------- дані ----------
with open("chunks.json", encoding="utf-8") as f:
    all_chunks = json.load(f)
emb_all = np.load("embeddings.npy")
pos = [i for i, c in enumerate(all_chunks) if c["has_pair"]]          # лише 72 парні статті (варіант V1)
chunks = [all_chunks[i] for i in pos]
emb = emb_all[pos].astype("float32")
ids = [f'{c["article_id"]}_{c["lang"]}_{c["chunk_index"]}' for c in chunks]
assert len(set(ids)) == len(ids), "ідентифікатори фрагментів не унікальні"

qsets = {}
for name, path in (("основний", "queries.json"), ("контрольний", "queries_holdout.json")):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            qsets[name] = json.load(f)

model = SentenceTransformer(MODEL)
log(f"chromadb {chromadb.__version__} | фрагментів: {len(chunks)} | розмірність: {emb.shape[1]}")

# ---------- 1. створення колекції ----------
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
client = chromadb.PersistentClient(path=DB_PATH)
t0 = time.time()
col = client.create_collection(
    COLLECTION,
    configuration={"hnsw": {"space": "cosine", "ef_construction": 200, "ef_search": 100, "max_neighbors": 32}},
)
bs = client.get_max_batch_size()
for s in range(0, len(chunks), bs):
    e = s + bs
    col.add(ids=ids[s:e],
            embeddings=emb[s:e].tolist(),
            documents=[c["text"] for c in chunks[s:e]],
            metadatas=[{"article_id": c["article_id"], "lang": c["lang"],
                        "title": c["title"], "chunk_index": c["chunk_index"]} for c in chunks[s:e]])
log(f"Колекцію '{COLLECTION}' створено й заповнено за {time.time() - t0:.1f} с; записів: {col.count()}")
log(f"Конфігурація індексу: {col.configuration}")

# ---------- 2. пошук: Chroma проти точного ----------
idx = {l: [i for i, c in enumerate(chunks) if c["lang"] == l] for l in ("uk", "en")}
E = {l: emb[idx[l]] for l in idx}


def article_rank(metas, targets):
    seen = []
    for m in metas:
        a = m["article_id"]
        if a not in seen:
            seen.append(a)
        if a in targets:
            return len(seen)
    return None


def metrics(ranks):
    n = len(ranks)
    rk = lambda k: sum(1 for x in ranks if x is not None and x <= k) / n
    return n, rk(1), rk(5), rk(10), sum(1 / x for x in ranks if x) / n


log("\n=== Пошук: Chroma (HNSW) проти точного пошуку NumPy ===")
log(f"{'набір':12s}{'сценарій':14s}{'метод':14s}{'n':>4s}{'R@1':>7s}{'R@5':>7s}{'R@10':>7s}{'MRR':>8s}")
for qname, queries in qsets.items():
    qv = model.encode(["query: " + q["query"] for q in queries], normalize_embeddings=True)
    for scen in ("та сама мова", "між мовами"):
        r_chroma, r_exact, same_rank, overlaps = [], [], 0, []
        for qi, q in enumerate(queries):
            if scen == "між мовами" and q.get("pair") == "a59":
                continue
            lang = q["lang"] if scen == "та сама мова" else ("en" if q["lang"] == "uk" else "uk")
            res = col.query(query_embeddings=[qv[qi].tolist()], n_results=min(N_RESULTS, len(idx[lang])),
                            where={"lang": lang}, include=["metadatas"])
            rc = article_rank(res["metadatas"][0], set(q["relevant"]))
            order = np.argsort(-(E[lang] @ qv[qi]), kind="stable")
            rx = article_rank([chunks[idx[lang][k]] for k in order], set(q["relevant"]))
            r_chroma.append(rc)
            r_exact.append(rx)
            same_rank += (rc == rx)
            top_c = set(res["ids"][0][:10])
            top_x = {ids[idx[lang][k]] for k in order[:10]}
            overlaps.append(len(top_c & top_x) / 10)
        for label, ranks in (("Chroma HNSW", r_chroma), ("точний NumPy", r_exact)):
            n, a, b, c, m = metrics(ranks)
            log(f"{qname:12s}{scen:14s}{label:14s}{n:4d}{a:7.2f}{b:7.2f}{c:7.2f}{m:8.3f}")
        log(f"{'':12s}{scen:14s}збіг рангу статті: {same_rank}/{len(r_chroma)}; "
            f"середній збіг топ-10 фрагментів з точним пошуком: {np.mean(overlaps):.3f}")

# ---------- 3. функціональні тести ----------
log("\n=== Функціональні тести ===")
tests = []
q0 = emb[0].tolist()


def run(name, expected, fn):
    try:
        actual = fn()
    except Exception as ex:
        actual = f"помилка {type(ex).__name__}: {str(ex)[:90]}"
    tests.append((name, expected, actual))


run("Фільтр lang=uk повертає лише українські фрагменти", "усі lang = uk",
    lambda: "усі lang = uk" if all(m["lang"] == "uk" for m in
            col.query(query_embeddings=[q0], n_results=20, where={"lang": "uk"}, include=["metadatas"])["metadatas"][0])
    else "знайдено інші мови")
run("Фільтр за неіснуючим значенням lang=de", "порожній результат",
    lambda: "порожній результат" if col.query(query_embeddings=[q0], n_results=5, where={"lang": "de"})["ids"][0] == []
    else "є результати")
run("Фільтр за article_id=18", "усі фрагменти зі статті 18",
    lambda: "усі фрагменти зі статті 18" if all(m["article_id"] == 18 for m in
            col.query(query_embeddings=[q0], n_results=5, where={"article_id": 18}, include=["metadatas"])["metadatas"][0])
    else "знайдено інші статті")
run("Додавання запису з уже наявним id", "без помилки; запис ігнорується",
    lambda: (lambda before, doc_before: (col.add(ids=[ids[0]], embeddings=[emb[1].tolist()], documents=["ДУБЛЬ"],
             metadatas=[chunks[0] and {"article_id": chunks[0]["article_id"], "lang": chunks[0]["lang"],
                                       "title": chunks[0]["title"], "chunk_index": chunks[0]["chunk_index"]}]),
             f"без помилки; записів {before}->{col.count()}; текст змінено: {col.get(ids=[ids[0]])['documents'][0] == 'ДУБЛЬ'}")[1]
             )(col.count(), col.get(ids=[ids[0]])["documents"][0]))
run("Додавання вектора неправильної розмірності", "помилка про розмірність",
    lambda: col.add(ids=["bad_dim"], embeddings=[[0.1] * 3], documents=["x"],
                    metadatas=[{"article_id": -1, "lang": "uk", "title": "x", "chunk_index": 0}]) or "без помилки")
run("Запит вектором неправильної розмірності", "помилка про розмірність",
    lambda: col.query(query_embeddings=[[0.1] * 3], n_results=2) and "без помилки")
run("Повторне відкриття бази після перезапуску клієнта", f"{col.count()} записів",
    lambda: f"{chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION).count()} записів")


def backup_test():
    if os.path.exists(BACKUP_PATH):
        shutil.rmtree(BACKUP_PATH)
    shutil.copytree(DB_PATH, BACKUP_PATH)
    n = chromadb.PersistentClient(path=BACKUP_PATH).get_collection(COLLECTION).count()
    return f"копію відкрито, {n} записів"


run("Резервна копія: копіювання папки бази й відкриття копії", f"{col.count()} записів", backup_test)

log(f"{'№':3s}| {'Перевірка':62s}| {'Очікуваний результат':36s}| Фактичний результат")
for i, (n_, e_, a_) in enumerate(tests, 1):
    log(f"{i:<3d}| {n_[:62]:62s}| {e_[:36]:36s}| {a_}")

with open("chroma_report.txt", "w", encoding="utf-8-sig") as f:
    f.write("\n".join(report))
print("\nЗвіт збережено у chroma_report.txt")
