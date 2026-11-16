# UnivFD: Universal Fake Detector (CLIP-ViT Foundation Engine)

**Developer**: Hitesh Kumar  
**Institution**: NIELIT Delhi / AIDA (Batch B8)  
**Academic Evaluator**: Dr. Shivlok Singh  
**Curriculum Unit**: Module 36 (Unsupervised Latent Clustering & Synthetic Image Forensics)  
**Model Backbone**: `openai/clip-vit-base-patch16` (Frozen Vision Transformer)  

---

## 📌 Executive Summary

Modern generative diffusion models (FLUX.1, Midjourney v6, Stable Diffusion 3.5, ByteDance SeaDream, DALL-E 3) generate hyper-realistic imagery that easily fools standard Convolutional Neural Networks (CNNs). Standard CNNs trained from scratch suffer from severe overfitting to specific resolutions or JPEG artifacts.

**UnivFD (Universal Fake Detector)** implements the research breakthrough published by **Ojha et al. (UC Berkeley & Adobe Research, CVPR 2023)**. By utilizing a **frozen CLIP Vision Transformer (ViT-B/16)** pre-trained on 400M internet image-text pairs, the system extracts high-level universal perceptual representations. A lightweight forensic classification probe maps subtle non-optical generative diffusion artifacts into a discriminative 512-dimensional manifold, achieving **97.48% ROC-AUC** across unseen generators.

---

## 📐 Academic Compliance: Module 36 Syllabus

To fulfill the **Module 36 (Unsupervised Learning & Clustering)** requirements:
1. **Unsupervised Latent Manifold**: Raw pixel clustering fails on deepfakes (~50% accuracy) because semantic content (objects, lighting, colors) dominates distance metrics.
2. **PCA Dimensionality Reduction**: The 512-dimensional CLIP feature space is projected into a 2D discriminative latent manifold via Principal Component Analysis (PCA).
3. **K-Means Clustering ($k=2$)**: Unsupervised K-Means discovers two mathematically isolated clusters representing **Authentic Camera Optics (Centroid 0)** versus **Generative Diffusion Latents (Centroid 1)**, with an inter-centroid Euclidean distance of **0.285 units** and a high silhouette separation score.

---

## 📊 Empirical Benchmark Results

Evaluated on a balanced test benchmark of **600 independent, unseen images** (300 Authentic Camera Captures + 300 Generative AI Captures):

| Metric | Score | Significance |
| :--- | :---: | :--- |
| **ROC-AUC** | **97.48%** | Exceptional separation between camera optics and synthetic diffusion |
| **Test Accuracy** | **92.83%** | High cross-generator generalization |
| **Precision** | **92.13%** | Minimizes false-accusation rate on authentic photography |
| **Recall** | **93.67%** | High sensitivity in catching synthetic diffusion noise |
| **F1 Score** | **92.89%** | Robust harmonic mean between precision and recall |
| **Inference Speed** | **~180 ms** | Real-time CPU inference (zero GPU required) |

### Test Confusion Matrix:
```
                 Predicted Real    Predicted AI
Actual Real           276               24
Actual AI              19              281
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Recommended: Create and activate a clean virtual environment:
  ```bash
  python -m venv .venv
  # Windows
  .venv\Scripts\activate
  # Linux/macOS
  source .venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Forensic Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📂 Project Structure

```
UnivFD_Capstone3/
├── app.py                  # Main Streamlit Dashboard (Dark Luxury Forensic UI)
├── univfd_bundle.pt        # Trained PyTorch Forensic Probe & K-Means Centroids
├── train_univfd.py         # Full Training Pipeline & Feature Extraction Code
├── requirements.txt        # Minimal Python Dependencies
├── README.md               # Technical Documentation & Viva Defense Guide
└── demo_samples/           # Curated Real & AI Test Images for Instant Testing
    ├── real/               # Authentic Camera Captures
    └── ai/                 # Synthetic Generative Captures
```

---

## 🎓 Viva Voce Key Talking Points for Evaluators

**Q1: Why use a frozen Vision Transformer instead of training a CNN from scratch?**
> *"CNNs trained from scratch on small datasets overfit to specific image resolutions or JPEG compression artifacts. UnivFD adopts the CVPR 2023 methodology: leveraging CLIP's frozen representations pre-trained on 400M images makes the forensic probe robust to unseen diffusion generators like FLUX and Midjourney."*

**Q2: How does this satisfy the Unsupervised Learning (Module 36) syllabus?**
> *"While supervised classifiers set fixed linear boundaries, our project projects the deep features into a 2D latent space and executes Unsupervised K-Means clustering (k=2). The clusters naturally segregate authentic camera optics from generative diffusion distributions with a clean silhouette score, proving natural geometric clustering without human labels at inference time."*

