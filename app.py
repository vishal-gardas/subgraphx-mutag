"""
app.py  —  SubgraphX MUTAG Explainer  (Streamlit Cloud ready)
Run locally:  streamlit run app.py
"""

import os, copy, random
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Anaconda OpenMP fix

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import to_networkx, remove_self_loops, subgraph as pyg_subgraph

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SubgraphX — MUTAG Explainer",
    page_icon="🧬",
    layout="wide",
)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

# Model file lives at repo root (committed to GitHub)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gin_best.pt")

ATOM_SYMBOLS = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}

# ── GIN model (must match what was saved) ─────────────────────────────────────
class GINClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, num_layers: int = 3,
                 dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden
            mlp = Sequential(Linear(in_ch, hidden), BatchNorm1d(hidden),
                              ReLU(), Linear(hidden, hidden))
            self.convs.append(GINConv(mlp, train_eps=True))
        self.classifier = Sequential(
            Linear(hidden, hidden // 2), ReLU(),
            torch.nn.Dropout(dropout), Linear(hidden // 2, num_classes),
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.classifier(x)


# ── Shapley-based subgraph scoring ───────────────────────────────────────────
def compute_shapley_scores(model, data, n_samples: int = 50):
    """
    Monte-Carlo Shapley approximation:
    Randomly mask out subsets of nodes and measure how the target-class
    probability changes.  Nodes that reliably help get positive scores.
    """
    model.eval()
    zeros = torch.zeros(data.num_nodes, dtype=torch.long)
    with torch.no_grad():
        full_prob = F.softmax(model(data.x.float(), data.edge_index, zeros), dim=-1)[0,
                               int(data.y.item())]

    scores = torch.zeros(data.num_nodes)
    for _ in range(n_samples):
        k = random.randint(max(1, int(.5 * data.num_nodes)),
                           max(1, int(.9 * data.num_nodes)))
        subset = torch.tensor(random.sample(range(data.num_nodes), k))
        sub_ei, _ = pyg_subgraph(subset, data.edge_index,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
        sub_batch = torch.zeros(len(subset), dtype=torch.long)
        with torch.no_grad():
            try:
                sub_prob = F.softmax(
                    model(data.x[subset].float(), sub_ei, sub_batch), dim=-1
                )[0, int(data.y.item())]
            except Exception:
                sub_prob = torch.tensor(0.0)
        delta = (sub_prob - full_prob).item()
        for n in subset.tolist():
            scores[n] += delta

    return scores / max(n_samples, 1)


def top_nodes(scores, data, fraction: float = 0.5):
    k = max(1, int(fraction * data.num_nodes))
    return scores.argsort(descending=True)[:k].tolist()


# ── Load dataset & model (cached across re-runs) ──────────────────────────────
@st.cache_resource(show_spinner="Loading dataset and model…")
def load_resources():
    dataset = TUDataset(root="./dataset/mutag/", name="MUTAG",
                        use_node_attr=True, use_edge_attr=True)
    model = GINClassifier(in_channels=dataset.num_features)
    model_loaded = False
    if os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            # Unwrap nested checkpoint formats:
            #   {"state_dict": {...}, "architecture": ..., ...}  ← gin_mutag_checkpoint.pt
            #   {"model_state_dict": {...}}                       ← alternative convention
            #   plain state_dict                                  ← bare torch.save(model.state_dict())
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]
                elif "model_state_dict" in ckpt:
                    ckpt = ckpt["model_state_dict"]
            model.load_state_dict(ckpt)
            model_loaded = True
        except Exception as e:
            st.warning(f"Could not load model weights: {e}")
    model.eval()
    return dataset, model, model_loaded



# ── Draw molecule ─────────────────────────────────────────────────────────────
def draw_molecule(data, highlight_nodes, true_lbl, pred_lbl, pred_prob):
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=SEED)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    colors = ["#F0A500" if n in highlight_nodes else "#3A7EBF" for n in G.nodes()]
    labels = {n: ATOM_SYMBOLS.get(int(data.x[n].argmax().item()), "?") for n in G.nodes()}

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888", width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=600, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                             font_color="white", font_size=9, font_weight="bold")

    lm = {0: "Non-mutagenic", 1: "Mutagenic"}
    ax.set_title(
        f"True: {lm[true_lbl]}   |   Predicted: {lm[pred_lbl]} ({pred_prob:.1%})\n"
        "Gold = SubgraphX explanation  •  Blue = rest of molecule",
        color="white", fontsize=10, pad=10,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("SubgraphX — MUTAG Molecular Explainer")
st.markdown(
    "**Pick a molecule** from the 188-graph MUTAG dataset. "
    "A trained **GIN** predicts its mutagenicity, then **SubgraphX** "
    "highlights the atoms that most influenced the prediction."
)

dataset, model, model_loaded = load_resources()

if not model_loaded:
    st.warning(
        "Trained weights not found (`gin_best.pt`). "
        "Predictions are from a randomly-initialised model. "
        "Run `SubGraphX_Complete.ipynb` → Section 9 to save the weights, "
        "then commit `gin_best.pt` to the repo."
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    mol_idx    = st.slider("Molecule index", 0, len(dataset) - 1, 0)
    top_k_pct  = st.slider("Explanation size (% of atoms)", 10, 90, 50, 10) / 100.0
    n_samples  = st.slider("Shapley samples (more = slower, more stable)",
                            10, 200, 50, 10)
    run_btn    = st.button("Run SubgraphX Explanation", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Model: GIN  |  Dataset: MUTAG (Debnath et al., 1991)")

# ── Molecule info ─────────────────────────────────────────────────────────────
data = copy.deepcopy(dataset[mol_idx])
data.edge_index, edge_attr = remove_self_loops(data.edge_index,
                                                getattr(data, "edge_attr", None))
if edge_attr is not None:
    data.edge_attr = edge_attr
true_label = int(data.y.item())

col1, col2, col3 = st.columns(3)
col1.metric("Molecule", f"#{mol_idx}")
col2.metric("Atoms (nodes)", data.num_nodes)
col3.metric("Bonds (edges)", data.edge_index.size(1) // 2)

# ── Prediction ────────────────────────────────────────────────────────────────
batch_z = torch.zeros(data.num_nodes, dtype=torch.long)
with torch.no_grad():
    probs      = F.softmax(model(data.x.float(), data.edge_index, batch_z), dim=-1)[0]
pred_label = int(probs.argmax())
pred_prob  = float(probs[pred_label])
lm         = {0: "Non-mutagenic", 1: "Mutagenic"}

col4, col5 = st.columns(2)
col4.metric("True label", lm[true_label])
col5.metric("Prediction",
            f"{lm[pred_label]} ({pred_prob:.1%})",
            delta="Correct" if pred_label == true_label else "Wrong",
            delta_color="normal" if pred_label == true_label else "inverse")

# ── Explanation ───────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Computing Shapley scores ({n_samples} samples)…"):
        scores     = compute_shapley_scores(model, data, n_samples=n_samples)
        highlighted = top_nodes(scores, data, fraction=top_k_pct)

    fig = draw_molecule(data, highlighted, true_label, pred_label, pred_prob)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Node Shapley Scores")
    import pandas as pd
    df = pd.DataFrame({
        "Node"         : list(range(data.num_nodes)),
        "Atom"         : [ATOM_SYMBOLS.get(int(data.x[n].argmax()), "?")
                          for n in range(data.num_nodes)],
        "Shapley score": [round(float(scores[n]), 5) for n in range(data.num_nodes)],
        "In explanation": ["Yes" if n in highlighted else "No"
                           for n in range(data.num_nodes)],
    }).sort_values("Shapley score", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True)
else:
    fig = draw_molecule(data, [], true_label, pred_label, pred_prob)
    st.pyplot(fig)
    plt.close(fig)
    st.info("Click **Run SubgraphX Explanation** in the sidebar to highlight the key atoms.")
