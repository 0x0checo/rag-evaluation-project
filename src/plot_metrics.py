from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("results/metrics_scifact_all.csv")

    metrics = ["ndcg@10", "recall@100", "map@10", "p@10"]
    for m in metrics:
        plt.figure()
        plt.bar(df["method"], df[m])
        plt.xticks(rotation=30, ha="right")
        plt.title(f"SciFact - {m}")
        plt.tight_layout()

        out_path = Path("results") / f"plot_scifact_{m}.png"
        plt.savefig(out_path, dpi=200)
        print("Saved:", out_path)

if __name__ == "__main__":
    main()

