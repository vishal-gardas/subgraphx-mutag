# 🧬 Explainable Molecular Property Prediction Using Graph Neural Networks with SubgraphX

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vishal-gardas-subgraphx-mutag.streamlit.app)

> **Author:** Vishal Gardas | **Reg No:** 23BAI0114 | VIT Bhopal University

---

## Overview

This project implements an end-to-end **explainable molecular property prediction** pipeline on the MUTAG dataset. A **Graph Isomorphism Network (GIN)** classifies molecules as mutagenic or non-mutagenic, and **SubgraphX** explains each prediction by identifying the atoms that most influenced the model's decision.

The full system is deployed as an interactive **Streamlit web app** — no code required to use it.

---

## Live Demo

**[Launch the App →](https://vishal-gardas-subgraphx-mutag.streamlit.app)**

Pick any of the 188 MUTAG molecules, get a GIN prediction, and run SubgraphX to highlight the key atoms (gold = explanation, blue = rest of molecule).

---

## What This Project Does

```
MUTAG Dataset (188 molecules)
        │
        ▼
  Data Preprocessing
  (self-loop removal, stratified 80/10/10 split, class-imbalance weighting)
        │
        ▼
  GIN Model Training
  (3 × GINConv + BatchNorm + ReLU → GlobalSumPool → Linear)
        │
        ▼
  SubgraphX Explainability
  (Monte Carlo Tree Search + Shapley value approximation)
        │
        ▼
  Streamlit Web App
  (Interactive molecule picker + prediction + explanation visualization)
```

---

## Project Structure

```
subgraphx-mutag/
├── app.py                          # Streamlit deployment app
├── SubGraphX_Complete.ipynb        # Full notebook (EDA → GIN → SubgraphX)
├── gin_best.pt                     # Trained GIN weights
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

## Dataset: MUTAG

| Property | Value |
|---|---|
| Total molecules | 188 |
| Mutagenic (class 1) | 125 (66.5%) |
| Non-mutagenic (class 0) | 63 (33.5%) |
| Node features | 7-dim one-hot (atom type: C, N, O, F, I, Cl, Br) |
| Edge features | 4-dim one-hot (bond type: aromatic, single, double, triple) |
| Molecule size | 10–28 atoms |

---

## Model: Graph Isomorphism Network (GIN)

GIN is theoretically the most expressive message-passing GNN, equivalent to the Weisfeiler-Lehman graph isomorphism test. This makes it ideal for distinguishing structurally different molecules.

**Architecture:**
```
GINConv 1  (7 → 64,  MLP: Linear → BatchNorm → ReLU → Linear)
GINConv 2  (64 → 64, MLP: Linear → BatchNorm → ReLU → Linear)
GINConv 3  (64 → 64, MLP: Linear → BatchNorm → ReLU → Linear)
GlobalSumPool
Linear     (64 → 2)
```

**Training:** Adam optimizer · lr=0.001 · weighted CrossEntropy · early stopping

**Results (~test set, 19 molecules):**

| Metric | Score |
|---|---|
| Accuracy | ~84% |
| ROC-AUC | ~0.91 |
| F1 (mutagenic) | ~85% |
| F1 (non-mutagenic) | ~83% |

---

## Explainability: SubgraphX

SubgraphX uses **Monte Carlo Tree Search (MCTS)** to find the connected subgraph that maximizes a **Shapley value** approximation of its contribution to the GIN prediction.

Unlike soft edge masks (GNNExplainer), SubgraphX returns **structurally connected subgraphs** that map directly onto chemically meaningful functional groups:

- 🔴 **Nitro groups (-NO₂)** → strongest mutagenic motif
- 🔴 **Amino groups (-NH₂)** → reactive with DNA after metabolic activation
- 🔴 **Aromatic ring systems** → associated with mutagenic PAHs

**Faithfulness metrics:**
- Fidelity+ ~0.31 (removing the explanation drops confidence by 31%)
- Fidelity− ~0.18 (explanation alone retains 82% of the model's confidence)

---

## Running Locally

### Prerequisites
- Python 3.8+
- Anaconda or pip environment with PyTorch and PyTorch Geometric

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the notebook
Open `SubGraphX_Complete.ipynb` in Jupyter and run all cells top-to-bottom. Section 9 saves the trained model to `gin_best.pt`.

### Launch the web app
```bash
streamlit run app.py
```
App will open at **http://localhost:8501**

---

## Dependencies

```
torch==2.14.0
torch_geometric==2.8.0
streamlit>=1.35.0
networkx>=3.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## Key References

- Xu et al. (2019) — [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) *(GIN)*
- Yuan et al. (2021) — [On Explainability of GNNs via Subgraph Explorations](https://arxiv.org/abs/2102.05152) *(SubgraphX)*
- Ying et al. (2019) — [GNNExplainer](https://arxiv.org/abs/1903.03894)
- Fey & Lenssen (2019) — [PyTorch Geometric](https://arxiv.org/abs/1903.02428)

---

## License

This project is submitted as an academic coursework project at VIT Bhopal University.
