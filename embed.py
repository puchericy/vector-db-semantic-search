import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"
PREFIX = "passage: "   # моделі e5 очікують префікс; для інших моделей став ""

with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer(MODEL)
print("Розмірність:", model.get_sentence_embedding_dimension())
print("Макс. довжина входу (токенів):", model.max_seq_length)

texts = [PREFIX + c["text"] for c in chunks]
start = time.time()
emb = model.encode(texts, batch_size=32, normalize_embeddings=True,
                   show_progress_bar=True)
print("Час обчислення, с:", round(time.time() - start, 1))

np.save("embeddings.npy", emb)
print("Форма масиву:", emb.shape)