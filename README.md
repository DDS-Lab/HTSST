# HTSST: Heterogeneous Tabular Self-Supervised Transformer

Official codebase for the paper:

> **Heterogeneous Tabular Self-Supervised Transformer for Asymmetric Data Regime in Engineering: Case Study on Bridge Seismic Vulnerability Assessment**
>
> Cameron Heng, Benedict Wong, Kota Murakami, Julia C. Lensing, Marc O. Eberhard, Jeffrey W. Berman, and John Y. Choe
>
> *IEEE International Conference on Data Mining (ICDM) 2026 — Applied Track*

---

## Overview

This repository provides the model implementations and public-source feature data for HTSST, a self-supervised transformer architecture for label-scarce heterogeneous tabular data. HTSST introduces a *trinity embedding* that tokenizes each tabular feature as the sum of a value projection, a learned feature identity embedding, and a modality type embedding, enabling a single transformer encoder to jointly process numerical, categorical, and text features.

The case study predicts the **Shear-Critical Column Index (SCCI)** — a measure of bridge column susceptibility to seismic shear failure — using 4,914 unlabeled bridge records from the National Bridge Inventory and supplementary geospatial datasets, with only 370 labeled targets available for fine-tuning.

---

## Repository Structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
└── package/
    ├── data/
    │   ├── categorical.csv        # Integer-encoded categorical features (59 cols)
    │   ├── numerical.csv          # Min-max normalized numerical features (36 cols)
    │   ├── text.csv               # BERT WordPiece token IDs (160 tokens/row)
    │   ├── total_dataset.csv      # Merged feature matrix (all modalities)
    │   └── metadata.json          # Column manifest
    └── models/
        ├── htsst_ssl_pipeline.ipynb       # HTSST (proposed)
        ├── htsst_notext.ipynb             # HTSST w/o text (ablation)
        ├── ft_transformer_baseline.ipynb  # FT-Transformer (text + pretraining) baseline
        └── scarf_baseline.ipynb           # SCARF-MLP contrastive baseline
```

> **Note:** Labeled SCCI data and trained model checkpoints are **not included** in this repository. Checkpoints are generated at runtime. All notebooks require `labeled_dataset.csv` to run (see [Data Access](#data-access)). Each notebook is self-contained and uses `SEED=42`.

---

## Models

### HTSST — `htsst_ssl_pipeline.ipynb`

The proposed model. Each feature is tokenized via the trinity embedding (value projection + feature identity + modality type) and processed by a 4-layer transformer encoder (8 heads, d_model=128, FFN=512). Pretrains on all 4,914 unlabeled rows using masked token reconstruction, then fine-tunes on 370 labeled rows with a dual-head architecture (point + quantile) and CQR conformalization.

- **Input:** CLS + 36 numeric + 54 categorical + 160 text = 251 tokens
- **Text:** Learned from scratch (29,603 vocab BERT WordPiece IDs, no pretrained BERT weights)
- **Params:** 4,797,725

### HTSST w/o Text — `htsst_notext.ipynb`

Ablation removing the text modality to isolate the contribution of textual information. Same architecture and training pipeline, with the input sequence reduced to 91 tokens (CLS + 36 numeric + 54 categorical).

- **Params:** 987,933

### FT-Transformer — `ft_transformer_baseline.ipynb`

Capacity-matched FT-Transformer baseline (Gorishniy et al., 2021) with BERT text features and masked pretraining. Since FT-Transformer does not have native text handling, text is processed via frozen `bert-base-uncased` mean-pooled embeddings reduced to 128 dims via PCA, then concatenated with numerical features. Uses per-feature linear tokenization for numerical and `nn.Embedding` for categorical features.

- **Input:** CLS + 164 numeric (36 NBI + 128 text PCA) + 54 categorical
- **Params:** 1,008,925

> **Note:** This notebook implements the best-performing FT-Transformer configuration we evaluated (text + pretraining). The paper's main results table (Table II) reports the canonical "as published" FT-Transformer (no text, no pretraining) for an out-of-the-box comparison.

### SCARF-MLP — `scarf_baseline.ipynb`

Contrastive self-supervised baseline (Bahri et al., ICLR 2022). Pretrains a 4-layer MLP encoder (hidden dim 256) using NT-Xent loss with 60% random feature corruption. Categorical features are one-hot encoded per the original paper. Encoder is fully unfrozen during fine-tuning.

- **Input:** 36 numeric + 54 categorical (one-hot, 365 cols) = 401 dims
- **Params:** 695,811

---

## Results

All models evaluated with 5-fold stratified cross-validation using out-of-fold (OOF) pooled predictions. CQR conformalization at 90% nominal coverage. Loss weights: pinball=2, MSE=1.

**Table II from paper (OOF R²):**

| Model | R² (MSE head) | R² (Quantile mid) |
|-------|:---:|:---:|
| **HTSST (proposed)** | **+0.430** | **+0.436** |
| HTSST w/o text | +0.287 | +0.346 |
| SCARF-MLP | -0.270 | -0.240 |
| FT-Transformer | -0.279 | -0.325 |

> **Reproducibility note:** The paper reports original unseeded results for consistency with the initial accepted manuscript's results. The seeded notebooks in this repository (SEED=42) produce slightly different results (e.g., HTSST: R²=+0.460/+0.427). The FT-Transformer row in the paper corresponds to the canonical "as published" configuration (no text, no pretraining); the included `ft_transformer_baseline.ipynb` runs the best-performing variant (text + pretraining, R²=-0.106/-0.087).

---

## Data

### Included (Unlabeled, Public-Source)

The `package/data/` folder contains 4,914 preprocessed bridge feature records derived from public datasets. No labeled targets are included.

| File | Description |
|------|-------------|
| `total_dataset.csv` | Merged feature matrix (all modalities) |
| `categorical.csv` | 59 categorical NBI/geospatial features, integer-encoded |
| `numerical.csv` | 36 continuous NBI/geospatial features, min-max normalized |
| `text.csv` | Text-type NBI fields tokenized to `bert-base-uncased` WordPiece IDs |
| `metadata.json` | Column names for numerical and categorical features |

### Source Datasets

| Dataset | Provider | Features |
|---------|----------|----------|
| [National Bridge Inventory (NBI)](https://www.fhwa.dot.gov/bridge/nbi.cfm) | FHWA / US DOT | Structural, geometric, condition attributes |
| [Macrostrat](https://macrostrat.org/) | University of Wisconsin | Geologic unit classification |
| [National Seismic Hazard Model (NSHM)](https://www.usgs.gov/programs/earthquake-hazards/hazard-assessment) | USGS | PGA, spectral acceleration |
| [USGS Design Maps](https://earthquake.usgs.gov/designmaps/) | USGS | Site-specific design spectral parameters |
| [National Flood Hazard Layer (NFHL)](https://www.fema.gov/flood-maps/national-flood-hazard-layer) | FEMA | Flood zone designation, base flood elevation |

### Data Access

Labeled SCCI records were collected in collaboration with **Washington State Department of Transportation (WSDOT)** and are subject to infrastructure security restrictions. They are not included in this repository. The authors have no authority to grant access on WSDOT's behalf.

---

## Setup

Developed and tested with Python 3.14, PyTorch 2.13, CUDA 13.2.

```bash
pip install -r requirements.txt
```

The notebooks use CPU-compatible PyTorch settings by default. A GPU is not required but will accelerate pretraining.

---

## Citation

Currently incomplete, will be updated after conference proceedings are published.

```bibtex
@inproceedings{heng2026htsst,
  title     = {Heterogeneous Tabular Self-Supervised Transformer for Asymmetric
               Data Regime in Engineering: Case Study on Bridge Seismic
               Vulnerability Assessment},
  author    = {Heng, Cameron and Wong, Benedict and Murakami, Kota and
               Lensing, Julia C. and Eberhard, Marc O. and Berman, Jeffrey W.
               and Choe, John Y.},
  year      = {2026}
}
```

---

## License

Code is released under the [MIT License](LICENSE). Data files in `package/data/` are derived from public government datasets and provided for research reproducibility; original source terms apply.
