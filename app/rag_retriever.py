import os
import pickle
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/processed_products.csv"
INDEX_PATH = "data/products_faiss.index"
METADATA_PATH = "data/products_metadata.pkl"


def build_vector_store():
    df = pd.read_csv(DATA_PATH)

    texts = df["text"].fillna("").tolist()
    ids = df["DSLD ID"].tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    metadata = [{"dsld_id": ids[i], "text": texts[i]} for i in range(len(texts))]
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


def load_vector_store():
    index = faiss.read_index(INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, model


def retrieve(query, top_k=3):
    index, metadata, model = load_vector_store()

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(metadata[idx])

    return results


if __name__ == "__main__":
    build_vector_store()

    sample_results = retrieve("protein supplement for muscle recovery", top_k=3)
    for i, result in enumerate(sample_results, 1):
        print(f"\nResult {i}:")
        print(result["text"][:500])