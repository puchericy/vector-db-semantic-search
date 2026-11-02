"""Перевірка контрольних запитів: збіг слів із текстом статті і чи стаття вже була в основному наборі.
Запуск: py check_overlap.py queries_holdout.json"""
import json
import re
import sys

QFILE = sys.argv[1] if len(sys.argv) > 1 else "queries_holdout.json"
STOP = set("що як для від між або чи та які який яка яке його цієї цих усього всі всіх "
           "where what how the and for with that this which are can its used use any such "
           "from into than rather over also between within".split())

with open("articles_uk_en.json", encoding="utf-8") as f:
    records = json.load(f)
with open(QFILE, encoding="utf-8") as f:
    holdout = json.load(f)
try:
    with open("queries.json", encoding="utf-8") as f:
        used = {a for d in json.load(f) for a in d["relevant"] + d.get("partial", [])}
except FileNotFoundError:
    used = set()


def stems(s):
    return {w[:5] for w in re.findall(r"\w+", s.lower()) if len(w) >= 4 and w not in STOP}


for q in holdout:
    qs = stems(q["query"])
    best = max(len(qs & stems(records[a]["text_" + q["lang"]])) / max(len(qs), 1) for a in q["relevant"])
    bucket = "легкий" if best >= 0.75 else "середній" if best >= 0.5 else "складний"
    seen = [a for a in q["relevant"] if a in used]
    note = f" | УЖЕ БУЛО в основному наборі: {seen}" if seen else ""
    print(f'{q["id"]} [{q["lang"]}] збіг слів {best:.2f} -> {bucket}{note}')
