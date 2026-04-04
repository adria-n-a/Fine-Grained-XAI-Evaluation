# Fine-Grained-XAI-Evaluation
Evaluating the faithfulness of LIME and LRP consensus maps on the CUB-200-2011 dataset.

# Objective Evaluation of XAI Methods in Fine-Grained Classification 🦅

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c.svg)
![Captum](https://img.shields.io/badge/Captum-LRP-brightgreen)
![LIME](https://img.shields.io/badge/LIME-Explainability-orange)

## Project Overview
Despite achieving high accuracy in image recognition, deep learning models often act as a "black box". Users frequently hesitate to trust the decisions these models make simply because the underlying mathematical reasoning is opaque. While visual Explainable AI (XAI) methods help build user trust by providing visual explanations, their reliability is traditionally only assessed qualitatively.

This project aims to move toward objective transparency by evaluating the mathematical faithfulness of XAI methods. Specifically, it explores whether creating an **aggregated consensus map** between LIME and LRP improves the objective faithfulness of the explanation compared to the individual base methods.

## Research Question
> *How does an aggregated consensus map combining LIME and LRP attributions compare to its individual base methods in terms of quantitative faithfulness, as measured by AOPC (Area over the MoRF Curve) pixel-deletion tests in fine-grained animal classification?* 

## Methodology

### 1. Dataset & Baseline Model
* **Dataset:** CUB-200-2011 (Caltech-UCSD Birds). This dataset provides a ground-truth "gold standard" with human-annotated part locations to verify if heatmaps focus on legitimate biological traits.
* **Model:** A pre-trained `ResNet-50` was fine-tuned on the CUB-200-2011 dataset to serve as a robust, high-performing baseline classifier.
* **Evaluation Subset:** To ensure rigorous scientific evaluation, XAI methods were only applied to a highly confident subset of the data (the top 3 most confident predictions across all 200 classes).

### 2. Base XAI Methods
* **LIME (Local Interpretable Model-agnostic Explanations):** A post-hoc method that explains predictions by segmenting images into super-pixels and perturbing them to observe changes in model confidence.
* **LRP (Layer-wise Relevance Propagation):** A model-specific approach that decomposes the final prediction score and redistributes it backwards through the network layers to produce a pixel-wise relevance map.

### 3. Proposed Evaluation: Consensus Maps & Pixel-Flipping
***Consensus Maps:** Min-Max normalization and element-wise mathematical aggregation are used to isolate features where both LIME and LRP agree, filtering out algorithm-specific noise.
* **Pixel-Flipping (AOPC):** Features are sorted by relevance and iteratively perturbed. The drop in classifier confidence is monitored to calculate the Area over the MoRF Curve (AOPC). A higher AOPC indicates a more faithful explanation.

## 📂 Project Structure
```text
├── Data/
│   └── confident_subset.csv       # Filtered dataset of top confident predictions
├── Notebooks/
│   ├── Day1_Baseline_ResNet.ipynb # Model fine-tuning and dataset preparation
│   ├── Day2_LIME_Explainer.ipynb  # LIME implementation & mask generation
│   ├── Day2_LRP_Explainer.ipynb   # LRP implementation via Captum
│   └── Day3_Consensus_Map.ipynb   # Normalization, Aggregation, and Pixel-Flipping
├── Heatmaps/
│   ├── LIME_Heatmaps/             # Saved .npy arrays for LIME masks
│   ├── LRP_Heatmaps/              # Saved .npy arrays for LRP attributions
│   └── Consensus_Maps/            # Aggregated .npy consensus maps
└── README.md
