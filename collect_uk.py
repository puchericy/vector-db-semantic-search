import json
import os
import sys
import time
import requests

HEADERS = {"User-Agent": "CourseProject/0.1 (student project; contact: YOUR_EMAIL)"}
CATEGORY_UK = "Категорія:Бази даних"
LIMIT = 200
OUT = sys.argv[1] if len(sys.argv) > 1 else "articles_uk_en.json"

if os.path.exists(OUT):
    sys.exit(f"Файл {OUT} уже існує (знімок корпусу). Щоб не перезаписати його, "
             f"вкажіть інше ім'я: py collect_uk.py articles_new.json")


def api(lang, params, tries=6):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {**params, "format": "json"}
    for attempt in range(tries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:  # забагато запитів: чекаємо і повторюємо
            wait = int(r.headers.get("Retry-After", 5 * (attempt + 1)))
            print(f"429, чекаю {wait} с...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        d = r.json()
        if "query" in d:
            return d
        time.sleep(2 * (attempt + 1))  # відповідь без 'query' (помилка API): повтор
    raise RuntimeError(f"Не вдалося отримати відповідь API: {params}")


data = api("uk", {"action": "query", "list": "categorymembers",
                  "cmtitle": CATEGORY_UK, "cmtype": "page", "cmlimit": LIMIT})
titles = [m["title"] for m in data["query"]["categorymembers"]]
print("Знайдено статей:", len(titles))

records = []
for title in titles:
    d = api("uk", {"action": "query", "titles": title, "prop": "extracts|langlinks",
                   "explaintext": 1, "lllang": "en"})
    page = next(iter(d["query"]["pages"].values()))
    links = page.get("langlinks", [])
    title_en = links[0].get("*") if links else None
    text_en = ""
    if title_en:
        d2 = api("en", {"action": "query", "titles": title_en, "prop": "extracts",
                        "explaintext": 1, "redirects": 1})
        page2 = next(iter(d2["query"]["pages"].values()))
        text_en = page2.get("extract", "")
        time.sleep(1.0)
    records.append({"title_uk": title, "text_uk": page.get("extract", ""),
                    "title_en": title_en, "text_en": text_en})
    time.sleep(1.0)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

pairs = [r for r in records if r["title_en"] and r["text_uk"] and r["text_en"]]
print("Пар з обома непорожніми текстами:", len(pairs))
