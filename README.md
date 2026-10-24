# 🛡️ AI-Driven Synthetic Media Detection & Authentication Engine
### *4th-Year Major / Capstone AI/ML & Digital Forensics Project*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA%20RTX%204050-76B900.svg?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Project Overview
The **AI-Driven Synthetic Media Detection & Authentication Engine** is an advanced computer vision and digital forensics system designed to detect and authenticate synthetic, AI-generated (e.g. Midjourney, Stable Diffusion, DALL-E, GANs) and manipulated media.

Unlike naive deepfake classifiers that output a single uncalibrated percentage, this engine operates as a **Multi-Evidence Forensic Engine** that couples deep convolutional feature extraction with physics-based signal processing:
1. **Deep Transfer Learning**: Vision backbone (**EfficientNet-B4 / ResNet-50**) with calibrated uncertainty estimation (Monte-Carlo Dropout).
2. **2D Fast Fourier Transform (FFT) Spectral Analysis**: Detection of high-frequency periodic grid spikes and upsampling checkerboard artifacts.
3. **Error Level Analysis (ELA)**: Analysis of localized JPEG compression discrepancy.
4. **Camera Sensor Noise Residuals (PRNU Proxy)**: Verification of natural optical sensor noise homogeneity.
5. **Grad-CAM Visual Explainability**: Gradient-weighted spatial activation maps highlighting exact manipulated regions.
6. **Robustness Benchmark Suite**: Real-world evaluation under social-media compression, blur, and resizing perturbations.

---

## 🏗️ System Architecture Pipeline

```
                                [ User Media Upload ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
      [ Deep Learning Core ]                          [ Physics & Forensic Signals ]
      • EfficientNet-B4 / ResNet-50                   • 2D FFT Spectral Log-Magnitude
      • MC-Dropout Uncertainty                        • Azimuthal Power Decay Profile
      • Feature Embeddings & Logits                   • Error Level Analysis (ELA 90%)
                 │                                    • Sensor Noise PRNU Residuals
                 │                                    • YCbCr / HSV Chrominance
                 └───────────────────────┬───────────────────────┘
                                         ▼
                           [ Grad-CAM Explainability ]
                           • Spatial Activation Heatmaps
                           • Generative Artifact Localization
                                         │
                                         ▼
                        [ Composite Authenticity Engine ]
                        • Multi-Evidence Fusion (0 - 100%)
                        • Severity Risk Matrix (Low / Med / High)
                        • Verdict & Authenticity Certificate
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
    [ Streamlit Web Dashboard ]                     [ Downloadable PDF Report ]
    • Real-time File Diagnostics                    • Official Forensic Certificate
    • Multi-Spectral Viewer                         • SHA-256 Media Hashing
    • Robustness Stress-Tester                      • Courtroom-Admissible Summary
```

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Activation
The project is built on Python 3.11 with PyTorch CUDA 12.4 acceleration:
```powershell
# Activate the virtual environment
.venv\Scripts\activate
```

### 2. Dataset Initialization & Download
Prepares training, validation, test splits and populates the sample gallery:
```powershell
python dataset_downloader.py
```

### 3. Model Training (GPU Accelerated)
Trains the transfer learning vision detector with mixed precision:
```powershell
python train.py --epochs 5 --batch-size 16 --model efficientnet_b4
```

### 4. Benchmark Evaluation
Generates the Confusion Matrix, ROC-AUC curve, Classification Report, and JSON summary:
```powershell
python evaluate.py
```

### 5. Robustness Stress-Testing
Evaluates model degradation against JPEG compression, Gaussian blur, downsampling, and noise:
```powershell
python test_robustness.py
```

### 6. Launch Interactive Web Dashboard
Launches the Streamlit application on local browser:
```powershell
streamlit run app.py
```

---

## 📊 Key Mathematical Formulations for Viva & Reports

### 1. Multi-Evidence Authenticity Score Fusion
The composite synthetic probability $S_{\text{synthetic}}$ is computed as:
$$S_{\text{synthetic}} = w_{\text{DL}} \cdot P_{\text{DL}} + w_{\text{FFT}} \cdot S_{\text{FFT}} + w_{\text{ELA}} \cdot S_{\text{ELA}} + w_{\text{Noise}} \cdot S_{\text{Noise}}$$
*Default Weights: $w_{\text{DL}} = 0.55$, $w_{\text{FFT}} = 0.20$, $w_{\text{ELA}} = 0.15$, $w_{\text{Noise}} = 0.10$.*

### 2. 2D Fast Fourier Transform (Spectral Domain)
For an image $I(x, y)$ of size $M \times N$:
$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x, y) e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$
The centered log-magnitude power spectrum is defined as:
$$S(u, v) = 20 \log\left(1 + |F(u, v)|\right)$$

### 3. Grad-CAM Spatial Explainability
Gradients of the predicted class score $y^c$ with respect to activation map $A^k$:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i, j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

---

## 📁 Repository Structure
```
Synthetic-Media-Detection/
├── .venv/                      # Python 3.11 virtual environment with PyTorch CUDA 12.4
├── dataset/                    # Real & Synthetic datasets (Train / Val / Test)
│   ├── train/
│   ├── val/
│   └── test/
├── models/                     # Checkpoints (best_synthetic_detector.pth)
├── reports/                    # Generated ROC curves, confusion matrices, PDF audit reports
├── sample_images/              # Curated test gallery for 1-click Streamlit testing
├── src/
│   ├── config.py               # Central hyperparameters & hardware paths
│   ├── models/
│   │   ├── detector.py         # EfficientNet-B4 / ResNet-50 architectures
│   │   └── gradcam.py          # Grad-CAM explainability engine
│   ├── forensics/
│   │   ├── fft_analysis.py     # 2D FFT & Azimuthal power decay
│   │   ├── ela_analysis.py     # Error Level Analysis (JPEG compression)
│   │   ├── noise_analysis.py   # High-frequency sensor noise & PRNU
│   │   └── color_analysis.py   # Chrominance distribution
│   ├── engine/
│   │   └── authenticator.py    # Multi-Evidence Authenticity Fusion
│   └── utils/
│       ├── dataset_loader.py   # DataLoaders & Augmentation pipeline
│       ├── metrics.py          # Accuracy, ROC-AUC, EER, F1-Score calculations
│       └── report_gen.py       # Courtroom-grade PDF audit report generator
├── app.py                      # Interactive Streamlit Web Dashboard
├── train.py                    # PyTorch GPU training script
├── evaluate.py                 # Evaluation & test benchmarking script
├── test_robustness.py          # Perturbation & robustness stress-test script
├── dataset_downloader.py       # Dataset initialization script
├── requirements.txt            # Python dependencies
└── README.md                   # Full academic documentation & viva manual
```

---

## 🎓 Academic Viva Questions & Answers

**Q1: Why not just use a standard CNN classifier?**  
*Answer:* Standard CNN classifiers overfit to the training generator (e.g. StyleGAN) and fail dramatically when presented with unseen diffusion models (e.g. Midjourney, FLUX) or compressed social media images. Combining deep features with physics-based signals (FFT spectral decay, PRNU sensor noise, and ELA) provides invariant physical evidence of manipulation.

**Q2: What is the significance of the 2D FFT in AI detection?**  
*Answer:* Generative models utilize transposed convolutions or patch embeddings during upsampling. These leave subtle periodic checkerboard artifacts in pixel space that manifest as distinct high-frequency spikes and deviation from natural $1/f$ power spectrum decay in Fourier space.

**Q3: How does Grad-CAM provide explainability?**  
*Answer:* Grad-CAM computes gradients of the target class score with respect to feature maps of the final convolutional layer. By performing global average pooling on these gradients and weighting the activation maps with ReLU, it produces a visual heatmap localizing the exact regions that caused the network to classify the media as synthetic.

**Q4: How does Error Level Analysis (ELA) work?**  
*Answer:* ELA resaves the image at a known uniform JPEG compression level (typically 90%) and calculates the amplified pixel-wise difference. Real, untouched camera captures have a uniform error level across the entire frame, whereas manipulated or AI-generated sections exhibit distinct compression error rates.
