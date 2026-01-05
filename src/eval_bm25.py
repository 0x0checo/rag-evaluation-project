import pickle
from pathlib import Path    
from tqdm import tqdm
import bm25s
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader 
from beir.retrieval.evaluation import EvaluateRetrieval

def build_index(doc):
    title  = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return (title + "\n" + text).strip()

def load_data(dataset="scifact", split="test"):
    data_folder = Path("data_raw") / dataset / dataset
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_folder)).load(split=split)
    return corpus, queries, qrels

def main():
    dataset = "scifact"
    split = "test"
    topk = 100

    # load dataset
    corpus, queries, qrels = load_data(dataset=dataset, split=split)

    # load bm25 index
    # with open(f"indexes/bm25_{dataset}.pkl", "rb") as f:
    #     obj = pickle.load(f) # convert to dict
    # doc_ids = obj["doc_ids"]
    # bm25 = obj["bm25"]

    doc_ids = list(corpus.keys())
    texts = [build_index(corpus[did]) for did in doc_ids]

    # 可选：加 stopwords="en" 往往更好
    corpus_tokens = bm25s.tokenize(texts, stopwords="en")
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)

    # prepare doc texts
    docid2idx = {did :i for i, did in enumerate(doc_ids)}

    # build results in BEIR format: {qid: {doc_id: score}}
    """
    bm25s 官方 README 的 quickstart 是：
    results, scores = retriever.retrieve(query_tokens, k=2)
    其中 results 默认返回的是 doc 的“ID/索引”（不是分数、也不是文档本身）；如果你想返回文档内容才传 corpus=...。
    你目前的写法混合了 corpus=doc_ids + 自己扁平化 token 的逻辑，容易导致“查询实际没生效/分数全 0 → 永远返回同一批最靠前的 doc”。
    """
    results = {}
    for qid, q in tqdm(queries.items(), desc=f"BM25 retrieving ({dataset})"):
        q_tokens = bm25s.tokenize(q, stopwords="en")
        # 兼容：有的版本 tokenize(单个字符串)会返回一维 list[int]
        # bm25s.retrieve 更稳的输入是 (n_queries, tokens) 的二维结构
        if len(q_tokens) > 0 and isinstance(q_tokens[0], int):
            q_tokens = [q_tokens]
        
        idxs, scores = bm25.retrieve(q_tokens, k=topk)

        # tolist + flatten
        idxs_list = idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)
        scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)

        # 结果通常是 shape (1, k)
        if len(idxs_list) > 0 and isinstance(idxs_list[0], list):
            idxs_list = idxs_list[0]
        if len(scores_list) > 0 and isinstance(scores_list[0], list):
            scores_list = scores_list[0]

        pairs = [(doc_ids[int(i)], float(s)) for i, s in zip(idxs_list, scores_list)]
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        results[qid] = dict(pairs)


    
    # evaluate
    evaluator = EvaluateRetrieval()
    k_values = [10, 100]
    # sanity check
    import random

    sample_qids = random.sample(list(queries.keys()), 3)
    for sqid in sample_qids:
        rel_docs = qrels.get(sqid, {})
        rel_list = list(rel_docs.keys())

        print("\n[Sanity] qid:", sqid)
        print("query:", queries[sqid])
        print("num relevant docs in qrels:", len(rel_list))

        if rel_list:
            rd = rel_list[0]
            print("example relevant doc_id from qrels:", rd, "type:", type(rd))
            print("exists in corpus keys?:", rd in corpus)
            print("exists in BM25 doc_ids?:", rd in docid2idx)

        # 看看你的 results 里 doc_id 长什么样
        top_docs = sorted(results[sqid].items(), key=lambda x: x[1], reverse=True)[:10]
        print("top-3 retrieved doc_ids:", [d for d, _ in top_docs[:3]])
        print("top-3 types:", [type(d) for d, _ in top_docs[:3]])

        hit = any(did in rel_docs for did, _ in top_docs)
        print("hit@10:", hit)

    metrics  = evaluator.evaluate(qrels, results, k_values)
    print("EVAL RETURN TYPE:", type(metrics))

    # print & save
    if isinstance(metrics, tuple):
        merged = {}
        for part in metrics:
            if isinstance(part, dict):
                merged.update(part)
        metrics_dict = merged
    else:
        metrics_dict = metrics

    print("METRICS KEYS:", sorted(metrics_dict.keys()))

    def pick(keys):
        for k in keys:
            if k in metrics_dict:
                return metrics_dict[k]
        return None

    out = {
    "dataset": dataset,
    "method": "bm25",
    "ndcg@10": pick(["NDCG@10"]),
    "recall@100": pick(["Recall@100"]),
    "map@10": pick(["MAP@10"]),
    "p@10": pick(["P@10"]),
    }
    print("\nBM25 metrics:", out)

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame([out]).to_csv("results/metrics_bm25_scifact.csv", index=False)
    print("Saved -> results/metrics_bm25_scifact.csv")

if __name__ == "__main__":
    main()
