# Project Brief: Dementia Classification with CNN–ViT Fusion

**Title:** Evaluating CNN–ViT Feature Fusion for Dementia Status Classification under Limited MRI Slice Budgets

## What is our project?

We will build an MRI research prototype that predicts **CDR = 0 versus CDR > 0** using a pretrained CNN, a Vision Transformer and their combined features. It predicts the dataset’s Clinical Dementia Rating group, not a confirmed Alzheimer’s diagnosis.

## Why are we doing this?

Combining models adds computation but may not consistently improve prediction. We will ask: **When is CNN–ViT fusion worth its additional cost, and how many MRI slices are needed?** This gives the semester project a clear experimental purpose.

## Dataset and approach

**Dataset:** [OASIS-1 brain MRI and clinical labels](https://sites.wustl.edu/oasisbrains/home/oasis-1/), obtained through the [official access process](https://sites.wustl.edu/oasisbrains/home/access/). Start with participants aged 60 or older, exclude missing labels and keep one session per person. Confirm usable class counts before training.

**Process:** MRI → fixed preprocessing → 4, 8 or 16 axial slices → frozen ResNet-18 and ViT-B/16 → saved slice features → mean pooling → CNN-only, ViT-only or fused features → logistic regression → participant-level prediction.

## What is different, and what is our novelty?

| Existing approach or standard practice | Our proposed emphasis |
|---|---|
| CNN–Transformer hybrids already exist | Test whether fusion earns its extra computation |
| Slice attention and multi-view fusion exist | Use simple pooling and controlled 4/8/16-slice comparisons |
| Pretraining, feature caching and participant splits are established | Use these to make a fair, reproducible semester study |

**Proposed contribution:** New experimental evidence about the accuracy–computation trade-off of frozen CNN–ViT fusion under a specific MRI protocol. This is **potential empirical novelty**, not a new architecture or a verified “first-ever” method. A broader literature review is needed before claiming uniqueness.

## Available papers

| Paper and link | What it contributes |
|---|---|
| [Early detection of Alzheimer’s disease progression stages using hybrid of CNN and transformer encoder models](https://www.nature.com/articles/s41598-025-01072-5) — Scientific Reports, 2025 | Hybrid classification reference; establishes that combining CNN and Transformer is already studied. |
| [Training ViT with Limited Data for Alzheimer’s Disease Classification: an Empirical Study](https://papers.miccai.org/miccai-2024/796-Paper2724.html) — MICCAI, 2024 | Explores training strategies for limited 3D MRI data; useful empirical-study reference. |
| [A multi-slice attention fusion and multi-view personalized fusion lightweight network for Alzheimer’s disease diagnosis](https://link.springer.com/article/10.1186/s12880-024-01429-8) — BMC Medical Imaging, 2024 | Prior learned slice and view fusion. |
| [A Computer Vision Hybrid Approach: CNN and Transformer Models for Accurate Alzheimer’s Detection from Brain MRI Scans](https://arxiv.org/abs/2601.15202) — arXiv preprint, 2026 | Further hybrid-model prior work; not presented here as peer-reviewed evidence. |

## Is it doable?

**Yes, as a roughly 10-week prototype**, assuming dataset access, basic Python skills and GPU availability. Frozen encoders and cached embeddings limit repeated training work. ViT extraction still needs to be benchmarked. Tools: PyTorch, NiBabel, scikit-learn and optionally Streamlit.

**Core evaluation:** Nine model/slice configurations on identical participant splits; ROC-AUC, balanced accuracy, sensitivity, specificity, confidence intervals, extraction time and memory. Include an age-only baseline. Keep test participants untouched during tuning.

**Optional extension:** Mild noise/blur tests after the core study works; these do not establish robustness across hospitals.

**Deliverables:** Reproducible pipeline, saved features, model comparison and runtime plots, a sample-MRI demo, and a report. Fusion need not win: the outcome is evidence about when it helps. No accuracy results or clinical validity are claimed yet.
