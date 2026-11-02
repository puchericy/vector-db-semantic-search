"""Схема векторної БД (еквівалент DDL для Chroma): створення колекції й перевірка записів.

Колекція kb_chunks зберігає сутності ФРАГМЕНТ і ЕМБЕДИНГ з ER-діаграми:
  id          -> ідентифікатор запису Chroma, формат "<article_id>_<lang>_<chunk_index>"
  embedding   -> вектор 384 float32, нормалізований (модель intfloat/multilingual-e5-small)
  document    -> text, текст фрагмента (ліміт розбиття 800 знаків, найдовший у корпусі 1209)
  metadata    -> article_id (int), lang ("uk"/"en"), title (str), chunk_index (int)
Сутності СТАТТЯ, ЗАПИТ ОЦІНЮВАННЯ, ВІДПОВІДЬ у Chroma не зберігаються: вони лежать у JSON-файлах
(chunks.json, queries.json, queries_holdout.json).
Запуск: py schema_chroma.py            (створює порожню колекцію й перевіряє схему на тестовому записі)
"""
import re

import chromadb

DB_PATH, COLLECTION, DIM = "chroma_db_schema_demo", "kb_chunks", 384
MAX_CHUNK_CHARS = 1209  # найдовший фрагмент у chunks.json (ліміт chunk.py 800; одне довге речення не ділиться)
HNSW = {"space": "cosine", "ef_construction": 200, "ef_search": 100, "max_neighbors": 32}
ID_RE = re.compile(r"^\d+_(uk|en)_\d+$")


def create_collection(client):
    """Створює колекцію з індексом HNSW (косинусна відстань)."""
    return client.create_collection(COLLECTION, configuration={"hnsw": HNSW})


def validate(record_id, embedding, document, metadata):
    """Перевірка запису перед додаванням (Chroma сама перевіряє лише розмірність вектора)."""
    errors = []
    if not ID_RE.match(record_id):
        errors.append("id має вигляд <article_id>_<uk|en>_<chunk_index>")
    if len(embedding) != DIM:
        errors.append(f"вектор має {len(embedding)} чисел, потрібно {DIM}")
    if not 0 < len(document) <= MAX_CHUNK_CHARS:
        errors.append(f"довжина тексту {len(document)} поза межами 1..{MAX_CHUNK_CHARS}")
    if set(metadata) != {"article_id", "lang", "title", "chunk_index"}:
        errors.append("метадані мають містити рівно article_id, lang, title, chunk_index")
    else:
        if not isinstance(metadata["article_id"], int) or not isinstance(metadata["chunk_index"], int):
            errors.append("article_id і chunk_index мають бути цілими")
        if metadata["lang"] not in ("uk", "en"):
            errors.append("lang має бути uk або en")
        if record_id.split("_")[:1] != [str(metadata["article_id"])]:
            errors.append("article_id у id і в метаданих мають збігатися")
    return errors


if __name__ == "__main__":
    client = chromadb.PersistentClient(path=DB_PATH)
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = create_collection(client)
    print("Колекцію створено:", col.name, "| конфігурація:", col.configuration)
    rec = ("1_uk_0", [0.1] * DIM, "Приклад тексту фрагмента.",
           {"article_id": 1, "lang": "uk", "title": "Приклад", "chunk_index": 0})
    print("Помилки схеми для коректного запису:", validate(*rec) or "немає")
    col.add(ids=[rec[0]], embeddings=[rec[1]], documents=[rec[2]], metadatas=[rec[3]])
    print("Записів у колекції:", col.count())
    bad = ("x", [0.1] * 3, "", {"lang": "de"})
    print("Помилки схеми для некоректного запису:", validate(*bad))
