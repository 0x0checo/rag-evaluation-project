import pickle
from pathlib import Path    
from tqdm import tqdm
import bm25s
from beir.datasets.data_loader import GenericDataLoader 

def build_text(doc):
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return (title + "\n" + text).strip()

def main():
    dataset = "scifact" 
    data_path = Path("data_raw") / dataset /dataset
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split="test")

    doc_ids = list(corpus.keys())
    texts = [build_text(corpus[did]) for did in tqdm(doc_ids, desc="Preparing docs")]

    bm25 = bm25s.BM25()
    # tokenize then index and retrieve
    corpus_tokens = bm25s.tokenize(texts)
    bm25.index(corpus_tokens)

    qid = next(iter(queries))
    q = queries[qid]
    q_tokens = bm25s.tokenize(q)
    if len(q_tokens) > 0 and isinstance(q_tokens[0], list):
        q_tokens = q_tokens[0]
    scores, idxs = bm25.retrieve(q_tokens, k=5)

    # scores/idxs 可能是 numpy array 或 shape(1,k) 的结构，先拍平
    try:
        scores_list = scores.tolist()
    except Exception:
        scores_list = list(scores)

    try:
        idxs_list = idxs.tolist()
    except Exception:
        idxs_list = list(idxs)

    # 如果是 [[...]] 这种结构，取第一层
    if len(idxs_list) > 0 and isinstance(idxs_list[0], list):
        idxs_list = idxs_list[0]
    if len(scores_list) > 0 and isinstance(scores_list[0], list):
        scores_list = scores_list[0]

    print("Top-5 results:")
    for rank, (s, i) in enumerate(zip(scores_list, idxs_list), start=1):
        i = int(i)
        did = doc_ids[i]
        snippet = texts[i][:200].replace("\n", " ")
        print(f"{rank:02d}. score={float(s):.4f} doc_id={did} snippet={snippet}")


    # Save indexes
    Path("indexes").mkdir(exist_ok=True)
    with open(f"indexes/bm25_{dataset}.pkl", "wb") as f:
        pickle.dump({"doc_ids": doc_ids, "bm25": bm25}, f)

    print("\nSaved BM25 index to:", f"indexes/bm25_{dataset}.pkl")

if __name__ == "__main__":
    main()
