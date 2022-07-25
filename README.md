# Exploring Deep Learning in Finance  
**Attention-Based Transformer for Robust Financial Time Series Classification**

---

## Overview

This Master's Thesis investigates the application of **Attention-Based Transformer Encoders** to build **robust Buy/Sell classification models** for financial time series data.  

Financial markets are inherently **non-stationary, noisy, and regime-dependent**, which challenges conventional machine learning models.  
To address these issues, this research integrates **De Prado-inspired data preprocessing** with a **hybrid Transformer-LSTM architecture**, enabling improved temporal modeling and robust evaluation.

### Key Contributions
- **De Prado–Inspired Data Pipeline**: Combines *Dollar-Bars* sampling with the *Triple-Barrier Method* for economically meaningful labeling.  
- **Hybrid Transformer–LSTM Model**: Fuses *long-range attention* with *short-term memory* for richer temporal understanding.  
- **Leak-Free Evaluation Framework**: Implements *Purged K-Fold Cross-Validation* to ensure realistic backtesting integrity.  

---

## Research Objectives

1. **Robust Data Structuring**: Construct an **information-dense, volatility-adjusted dataset** using Lopez de Prado’s techniques.  
2. **Advanced Temporal Modeling**: Develop a **Transformer Encoder–based architecture** to learn **multi-scale, non-linear dependencies**.  
3. **Rigorous Validation**: Apply **Purged K-Fold Cross-Validation** to avoid temporal leakage and ensure true out-of-sample generalization.  

---

# Transformer Encoder Block

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*7sjcgd_nyODdLbZSxyxz_g.png" alt="Transformer Encoder Block Diagram">
</p>

*Source: [ResearchGate](https://www.researchgate.net/figure/Block-diagram-of-the-transformer-encoder_fig6_349339665)*

The Transformer Encoder is a fundamental component of the Transformer architecture, introduced in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. It generates contextualized representations for downstream tasks.

---

## Structure Overview

Each encoder block consists of:

1. **Multi-Head Self-Attention (MHSA)**  
2. **Add & Norm (Residual Connection + Layer Normalization)**  
3. **Position-Wise Feed-Forward Network (FFN)**  
4. **Add & Norm (Residual Connection + Layer Normalization)**  

Typically stacked 6+ times to form the complete encoder.

---

## Detailed Components

### 1. Multi-Head Self-Attention (MHSA)

- **Purpose:** Capture dependencies between all tokens in the sequence.
- **Mechanism:** For each token, compute Query (Q), Key (K), and Value (V) vectors. Attention is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

- **Multi-Heading:** Uses multiple attention heads to capture different relational aspects.

### 2. Add & Norm

- **Residual Connection:** Input + sub-layer output to aid gradient flow.  
- **Layer Normalization:** Stabilizes training across features.

### 3. Position-Wise Feed-Forward Network (FFN)

$$
\text{FFN}(x) = \text{max}(0, x W_1 + b_1) W_2 + b_2
$$

- **Activation:** ReLU or GELU.  
- **Function:** Enhances token-wise representation independently.

### 4. Add & Norm (FFN output)

- Residual + normalization, as before.

---

## Stacking Encoder Layers

- 6+ identical layers  
- Outputs of one layer feed as input to the next  
- Enables progressively abstract sequence representations

---

## Positional Encoding

Since Transformers lack intrinsic order, positional encodings are added:

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

---

## Advantages

- **Parallelization**: Faster than RNNs  
- **Long-Range Dependencies**: Captures distant relationships  
- **Scalability**: More layers/heads improve performance

---

## Further Reading

- [Attention Is All You Need (Paper)](https://arxiv.org/abs/1706.03762)  
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)  
- [Transformer Explained Visually](https://poloclub.github.io/transformer-explainer/)

---

## Results and Performance Analysis

### 1. Experimental Setup

| Parameter | Description |
|------------|-------------|
| **Dataset** | Multi-asset OHLCV converted to Dollar-Bars |
| **Feature Set** | Technical indicators (RSI, MACD, Bollinger Bands), PCA-reduced |
| **Labeling** | Triple-Barrier Method |
| **Validation** | Purged K-Fold (5 folds) |
| **Framework** | PyTorch |

### 2. Model Comparison

| Model | Mean Balanced Accuracy | Mean AUC Score |
|--------|------------------------|----------------|
| **Transformer–LSTM (Proposed)** | 🟢 62.1% | 🟢 0.65 |
| Random Forest | 55.4% | 0.58 |
| SVM | 52.8% | 0.53 |
| Logistic Regression | 50.1% | 0.50 |

### 3. Insights & Discussion

- Attention layers captured **latent inter-bar dependencies**  
- LSTM hybridization enhanced **short-term recall**  
- False positives occurred during **high-volatility periods**  
- Hybrid attention + Dollar-Bars/TBM improves **robustness and interpretability**

---

## 🔭 Future Directions

- Temporal Fusion Transformers for multi-horizon forecasting  
- Volatility clustering features (GARCH, realized volatility)  
- Adaptive fine-tuning for regime-aware updates

---

## Conclusion

- Attention-based architectures improve **predictive power and robustness** in financial time series  
- Data preprocessing (Dollar-Bars + TBM) ensures **economically meaningful labeling**  
- Hybrid Transformer–LSTM captures **macro and micro temporal patterns**  

---

**Back to Main:** [README.md](./README.md)  
**Figures & Results:** [`figures/`](./figures)  
**Thesis Report:** [`thesis_document.pdf`](https://mavmatrix.uta.edu/industrialmanusys_theses/18/?utm_source=mavmatrix.uta.edu%2Findustrialmanusys_theses%2F18&utm_medium=PDF&utm_campaign=PDFCoverPages)
