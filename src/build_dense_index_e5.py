import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm

def build_text(doc):
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return (title + "\n" + text).strip()

def load_data_folder(dataset="scifact"):
    return Path("data_raw") / dataset / dataset

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def main():
    dataset = "scifact"
    model_name = "intfloat/e5-base-v2"
    tag = "e5"

    data_folder = load_data_folder(dataset=dataset)
    corpus, _, _ = GenericDataLoader(data_folder=str(data_folder)).load(split="test")

    doc_ids = list(corpus.keys())
    texts = [build_text(corpus[did]) for did in doc_ids]
    # add passage prefix
    texts = ["passage: " + t for t in texts]

    print("Loading model:", model_name)
    model = SentenceTransformer(model_name)

    # encode
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    vecs = l2_normalize(vecs.astype("float32"))

    # FAISS cosine via inner product on normalized vectors
    # 1. Get the dimensionality of the embedding vectors
    dim = vecs.shape[1]
    # 2. Initialize a flat FAISS index using Inner Product (IP) similarity
    index = faiss.IndexFlatIP(dim)
    # 3. Add the normalized document embeddings to the index for retrieval
    index.add(vecs)

    # keep indexes
    Path("indexes").mkdir(exist_ok=True)
    faiss.write_index(index, f"indexes/faiss_{dataset}_{tag}.index")
    with open(f"indexes/faiss_{dataset}_{tag}_docids.json", "w") as f:
        json.dump(doc_ids, f)

    print("Saved index ->", f"indexes/faiss_{dataset}_{tag}.index")


if __name__ == "__main__":
    main()