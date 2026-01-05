import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import numpy as np
import bm25s
from pathlib import Path    
from sentence_transformers import SentenceTransformer
import faiss
faiss.omp_set_num_threads(1)
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

def build_index(doc):
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return (title + "\n" + text).strip()

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def rrf_fuse(rank_lists, k=60):
    scores = {}
    for docs in rank_lists:
        for r, did in enumerate(docs, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k+r)
    return scores

def load_data(dataset="scifact", split="test"):
    data_folder = Path("data_raw") / dataset /dataset
    return GenericDataLoader(data_folder=str(data_folder)).load(split="test")

def main():
    dataset = "scifact"
    split = "test"
    topk = 100
    fuse_k = 60

    # ---------- Load dataset ----------
    corpus, queries, qrels = load_data(dataset, split)

    # ---------- Build BM25 on-the-fly ----------
    doc_ids = list(corpus.keys())
    texts = [build_index(corpus[did]) for did in doc_ids]
    corpus_tokens = bm25s.tokenize(texts=texts, stopwords="en")
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)

    # ---------- Load BGE FAISS index ----------
    model_name = "BAAI/bge-small-en-v1.5"
    tag = "bge"
    index = faiss.read_index(f"indexes/faiss_{dataset}_{tag}.index")
    with open(f"indexes/faiss_{dataset}_{tag}_docids.json") as f:
        dense_doc_ids = json.load(f)

    assert set(dense_doc_ids) == set(doc_ids), "Dense doc_ids mismatch corpus keys!"
    
    model = SentenceTransformer(model_name)

    # ---------- Hybrid retrieval ----------
    results = {}
    qids = list(queries.keys())
    qtexts = [queries[qid] for qid in qids]

    # Dense search in batches (fast)
    qvecs = model.encode(qtexts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    qvecs = l2_normalize(qvecs.astype("float32"))
    qvecs = np.ascontiguousarray(qvecs, dtype="float32")
    dense_scores, dense_idxs = index.search(qvecs, topk)

    for qi, qid in enumerate(tqdm(qids, desc="Hybrid (BM25+BGE)")):
        q = queries[qid]

        # BM25 ranking list  
        q_tokens = bm25s.tokenize(q, stopwords="en")
        if len(q_tokens) > 0 and isinstance(q_tokens[0], int):
            q_tokens = [q_tokens]
        bm25_idxs, bm25_scores = bm25.retrieve(q_tokens, k=topk)
        bm25_idxs = bm25_idxs.tolist()[0] if hasattr(bm25_idxs, "tolist") else list(bm25_idxs)[0]
        bm25_rank = [doc_ids[int(i)] for i in bm25_idxs]

        # Dense ranking list (doc_id)
        dids = [dense_doc_ids[int(i)] for i in dense_idxs[qi]]
        dense_rank = dids

        # RRF fuse
        fused = rrf_fuse([bm25_rank, dense_rank], k=fuse_k)
        fused_top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:topk]
        results[qid] = dict(fused_top)
    
    # ---------- Evaluate ----------
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, results, [10, 100])

    # print output
    if isinstance(metrics, tuple):
        merged = {}
        for part in metrics:
            if isinstance(part, dict):
                merged.update(part)
        metrics = merged

    out = {
        "dataset": dataset,
        "method": "hybrid_rrf_bm25+bge",
        "ndcg@10": metrics["NDCG@10"],
        "recall@100": metrics["Recall@100"],
        "map@10": metrics["MAP@10"],
        "p@10": metrics["P@10"],
    }

    print("\nHybrid metrics:", out)

if __name__ == "__main__":
    main()
