from beir.util import download_and_unzip
from pathlib import Path    

def main():
    dataset = "scifact"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = Path("data_raw") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print("downloading:", url)
    download_and_unzip(url, str(out_dir))
    print("Done! Save to:", out_dir)

if __name__ == "__main__":
    main()