# RAG Retrieval Engine: Sparse/Dense/Hybrid Baselines + Evaluation

A lightweight RAG retrieval benchmark + demo app built on **BEIR (SciFact)**.  
It compares **sparse retrieval (BM25)**, **dense retrieval (BGE / E5)**, and a **hybrid fusion (BM25 + BGE via RRF)**, with standard IR metrics and an interactive **Streamlit** UI.

---

## Features

- **Retrievers**
  - **BM25** (keyword/sparse)
  - **BGE** dense retrieval (vector)
  - **E5** dense retrieval (vector; uses `query:` prefix)
  - **Hybrid**: BM25 + BGE with **RRF (Reciprocal Rank Fusion)**

- **Evaluation (BEIR)**
  - Metrics: `NDCG@10`, `MAP@10`, `P@10`, `Recall@100`
  - Output CSV summary for easy reporting / plotting

- **Interactive Demo (Streamlit)**
  - Switch retriever from a dropdown
  - Choose a SciFact query ID or type your own query
  - View top-k retrieved docs with snippet/full text
  - Show **qrels ground truth** and **hit@k**

---

## Project Structure

