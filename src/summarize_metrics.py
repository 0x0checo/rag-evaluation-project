from pathlib import Path
import pandas as pd

def main():
    rows = [
        {"dataset": "scifact", "method": "bm25",
         "ndcg@10": 0.66169, "recall@100": 0.87589, "map@10": 0.61988, "p@10": 0.08600},

        {"dataset": "scifact", "method": "dense_bge",
         "ndcg@10": 0.72000, "recall@100": 0.95333, "map@10": 0.67636, "p@10": 0.09533},

        {"dataset": "scifact", "method": "dense_e5",
         "ndcg@10": 0.71943, "recall@100": 0.96267, "map@10": 0.67307, "p@10": 0.09533},

        {"dataset": "scifact", "method": "hybrid_rrf_bm25+bge",
         "ndcg@10": 0.71060, "recall@100": 0.96500, "map@10": 0.66770, "p@10": 0.09367},
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values(by="ndcg@10", ascending=False)

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / "metrics_scifact_all.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # print table
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

