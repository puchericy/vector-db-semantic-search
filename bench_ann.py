"""Швидкість і точність пошуку: точний перебір (NumPy) проти HNSW у Chroma на згенерованих даних.

Дані: реальні ембединги з embeddings.npy + гауссів шум, нормалізовані (SIGMA). Це НЕ реальні тексти:
результати швидкості показують тенденцію, а не продуктивність на справжньому корпусі.
Метрики якості: Recall@10 (id) = частка збігу номерів з точним топ-10; «Схожість до точного» = середня косинусна
схожість знайдених 10 до середньої схожості точного топ-10 (1,000 = так само добрі сусіди, навіть якщо номери інші).
Запуск: py bench_ann.py                   (обсяги 10 000, 100 000 і 300 000)
        py bench_ann.py 100000 1000000   (обсяг 10^6 потребує багато пам'яті й часу)
        py bench_ann.py 5000 20000       (швидка перевірка)
Результат: bench_report.txt, bench_results.csv
"""
import csv
import os
import platform
import shutil
import statistics
import sys
import time

import chromadb
import numpy as np

SIZES = [int(x) for x in sys.argv[1:]] or [10_000, 100_000, 300_000]
N_QUERIES, K, RUNS, SIGMA, SEED = 200, 10, 5, 0.15, 42
DB = "chroma_bench"

report, rows = [], []


def log(s=""):
    print(s)
    report.append(s)


base = np.load("embeddings.npy").astype("float32")
dim = base.shape[1]


def synth(n, seed):
    rng = np.random.RandomState(seed)
    x = base[rng.randint(0, len(base), n)] + SIGMA * rng.randn(n, dim).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def topk_exact(E, q, k, mask=None):
    sc = E @ q
    if mask is not None:
        sc = np.where(mask, sc, -np.inf)
    part = np.argpartition(-sc, k)[:k]
    return part[np.argsort(-sc[part])]


log(f"Python {platform.python_version()} | chromadb {chromadb.__version__} | процесор: {platform.processor()} | "
    f"логічних ядер: {os.cpu_count()}")
log(f"Параметри: запитів {N_QUERIES}, k={K}, прогонів {RUNS}, шум {SIGMA}, розмірність {dim}, "
    "HNSW: cosine, ef_construction 200, ef_search 100, max_neighbors 32")
log("Дані згенеровані з реальних ембедингів додаванням шуму; половина записів має lang=uk, половина lang=en.\n")

if os.path.exists(DB):
    shutil.rmtree(DB, ignore_errors=True)
client = chromadb.PersistentClient(path=DB)
Q = synth(N_QUERIES, SEED + 1)

log(f"{'N':>9s} | {'режим':14s} | {'метод':12s} | {'побудова, с':>11s} | {'затримка запиту, мс (середнє ± σ за прогонами)':>46s} | Recall@{K} (id) | Схожість до точного")
for n in SIZES:
    E = synth(n, SEED)
    lang = np.where(np.random.RandomState(SEED + 2).rand(n) < 0.5, "uk", "en")
    mask_uk = lang == "uk"
    col = client.create_collection(
        f"bench_{n}", configuration={"hnsw": {"space": "cosine", "ef_construction": 200, "ef_search": 100, "max_neighbors": 32}})
    bs = client.get_max_batch_size()
    t0 = time.time()
    for s in range(0, n, bs):
        e = min(s + bs, n)
        col.add(ids=[str(i) for i in range(s, e)], embeddings=E[s:e].tolist(),
                metadatas=[{"lang": str(l)} for l in lang[s:e]])
    build = time.time() - t0

    for mode, mask, where in (("без фільтра", None, None), ("lang = uk", mask_uk, {"lang": "uk"})):
        # точний пошук
        times = []
        for _ in range(RUNS):
            t = time.perf_counter()
            for q in Q:
                topk_exact(E, q, K, mask)
            times.append((time.perf_counter() - t) / N_QUERIES * 1000)
        ex_top = [topk_exact(E, q, K, mask) for q in Q]
        ex_ids = [set(t.tolist()) for t in ex_top]
        ex_sim = [float((E[t] @ q).mean()) for t, q in zip(ex_top, Q)]
        log(f"{n:>9d} | {mode:14s} | {'точний NumPy':12s} | {'-':>11s} | "
            f"{statistics.mean(times):>10.2f} ± {statistics.pstdev(times):.2f}{'':>26s} | 1.000 (еталон)     | 1.000")
        rows.append({"N": n, "mode": mode, "method": "exact", "build_s": "", "ms_mean": statistics.mean(times),
                     "ms_sd": statistics.pstdev(times), "recall": 1.0, "sim_ratio": 1.0})
        # Chroma HNSW
        col.query(query_embeddings=[Q[0].tolist()], n_results=K, where=where)   # прогрів
        times = []
        for _ in range(RUNS):
            t = time.perf_counter()
            for q in Q:
                col.query(query_embeddings=[q.tolist()], n_results=K, where=where)
            times.append((time.perf_counter() - t) / N_QUERIES * 1000)
        rec, simr = [], []
        for q, ex, es in zip(Q, ex_ids, ex_sim):
            got_l = [int(i) for i in col.query(query_embeddings=[q.tolist()], n_results=K, where=where)["ids"][0]]
            rec.append(len(set(got_l) & ex) / K)
            simr.append(float((E[got_l] @ q).mean()) / es if got_l else 0.0)
        log(f"{n:>9d} | {mode:14s} | {'Chroma HNSW':12s} | {build:>11.1f} | "
            f"{statistics.mean(times):>10.2f} ± {statistics.pstdev(times):.2f}{'':>26s} | {statistics.mean(rec):.3f}              | {statistics.mean(simr):.3f}")
        rows.append({"N": n, "mode": mode, "method": "chroma_hnsw", "build_s": round(build, 1),
                     "ms_mean": statistics.mean(times), "ms_sd": statistics.pstdev(times), "recall": statistics.mean(rec),
                     "sim_ratio": statistics.mean(simr)})

with open("bench_report.txt", "w", encoding="utf-8-sig") as f:
    f.write("\n".join(report))
with open("bench_results.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
shutil.rmtree(DB, ignore_errors=True)
print("\nЗвіт: bench_report.txt, bench_results.csv")
