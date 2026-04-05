# Hybrid Differential Privacy Framework for NLP

A privacy-preserving architecture combining selective anonymization and sensitivity-aware DP-SGD, demonstrated on mental health conversational AI.

---

## Overview

Modern NLP systems are vulnerable to membership inference attacks, training data memorization, and PII leakage. Existing mitigations each fall short:

- **Anonymization** — no formal privacy guarantees
- **Standard DP-SGD** — strong guarantees, but significant utility loss

This project introduces a **hybrid approach** that integrates both, using a sensitivity scoring mechanism to allocate noise adaptively per input — preserving utility where risk is low, strengthening protection where it's high.

---

## Key Results

| Metric | Result |
|---|---|
| Memorization rate (DP models) | ~0% |
| MIA AUC | ≈ 0.5 (near-random) |
| Utility retained vs. baseline | ~79.5% |
| Best privacy–utility balance | V3 (Hybrid) |

---

## System Architecture

![System Architecture](Images/system_architecture.png)

---

## Framework Pipeline

1. **Preprocessing** — Text cleaning and normalization on GoEmotions (multi-label emotion dataset)
2. **Selective anonymization** — Masking URLs, numeric identifiers, and sensitive patterns
3. **Sensitivity scoring** — Per-utterance score based on emotion intensity and length
4. **DP-SGD training** — Gradient clipping + Gaussian noise injection with formal (ε, δ) accounting via Opacus
5. **Adaptive noise allocation** — Noise magnitude scaled by sensitivity score
6. **Evaluation** — Utility (macro-F1, precision, recall) and privacy (MIA, memorization, canary exposure)

![Research Pipeline](Images/research_pipeline.png)

---

## Sensitivity-Aware Mechanism

Dynamic noise adjustment based on per-input sensitivity scores, balancing protection strength against utility retention.

![Sensitivity Mechanism](Images/sensitivity_mechanism.png)

---

## Model Variants

| Model | Description |
|---|---|
| V0 | Baseline — no privacy, full utility |
| V1 | Anonymization only — no formal guarantees |
| V2 | Uniform DP-SGD — formal guarantees, higher utility cost |
| **V3** | **Hybrid (proposed) — best privacy–utility trade-off** |

---

## Project Structure
Hybrid-DP-NLP-Privacy/
│
├── README.md
├── requirements.txt
├── demo_app.py                  # Chatbot demo (Gradio)
│
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Cleaned dataset
│
├── models/
│   └── rq1_variants/            # Trained models (V0–V3)
│
├── models_rq2/                  # Checkpoints for RQ2 experiments
│
├── reports/
│   ├── rq1/                     # RQ1 results, CSVs, plots
│   └── demo_*.png               # Demo visualizations
│
├── src/
│   ├── data/                    # Preprocessing notebooks
│   ├── models/                  # RQ1–RQ4 notebooks
│   │   ├── 01_rq1_anonym_dp_sgd.ipynb
│   │   ├── 02_rq2_sensitivity_dp_sgd.ipynb
│   │   └── 03_rq3_mia_memorization_defense.ipynb
│   ├── utils/                   # Utility functions
│   └── images/                  # Research diagrams
│
└── Images/                      # Figures used in README/thesis
├── system_architecture.png
├── research_pipeline.png
└── sensitivity_mechanism.png

---

## Installation
```bash
git clone https://github.com/ThenumiSedara/Hybrid-DP-NLP-Privacy.git
cd Hybrid-DP-Approach-For-Mental-Health-Chatbots
pip install -r requirements.txt
```

## Running Experiments

| Research Question | Notebook / Command |
|---|---|
| RQ1: Hybrid vs. baselines | `src/models/01_rq1_anonym_dp_sgd.ipynb` |
| RQ2: Sensitivity-aware DP-SGD | `src/models/02_rq2_sensitivity_dp_sgd.ipynb` |
| RQ3: MIA + memorization attacks | `src/models/03_rq3_mia_memorization_defense.ipynb` |
| RQ4: Chatbot demo | `python demo_app.py` |

---

## Technologies

Python 3.10 · PyTorch · Hugging Face Transformers · Opacus · Gradio
