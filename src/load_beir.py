from beir.datasets.data_loader import GenericDataLoader 
from pathlib import Path

def main():
    dataset = "scifact"
    data_path = Path("data_raw") / dataset / dataset
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split="test")

    print("Dataset:", dataset)
    print("Docs:", len(corpus))
    print("Queries:", len(queries))
    print("Qrels:", len(qrels))

    any_doc_id = next(iter(corpus))
    any_qid = next(iter(queries))
    print("\nSample doc_id:", any_doc_id)
    print("Sample doc:", corpus[any_doc_id].keys())
    print("Sample query_id:", any_qid)
    print("Sample query:", queries[any_qid][:200])

if __name__ == "__main__":
    main()