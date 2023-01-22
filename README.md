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

## Tech Stack & Tools

<table>
  <tr>
    <td><strong>ML / DL Models</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Transformer--LSTM-blueviolet" alt="Transformer-LSTM">
      <img src="https://img.shields.io/badge/Random%20Forest-FF9900" alt="Random Forest">
      <img src="https://img.shields.io/badge/Logistic%20Regression-6CC644" alt="Logistic Regression">
      <img src="https://img.shields.io/badge/SVM-1E90FF" alt="SVM">
    </td>
  </tr>
  <tr>
    <td><strong>Programming & Frameworks</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Python-3.8%2B-green" alt="Python Version">
      <img src="https://img.shields.io/badge/PyTorch-red" alt="PyTorch Framework">
      <img src="https://img.shields.io/badge/Numpy-F5DEB3" alt="NumPy">
      <img src="https://img.shields.io/badge/Pandas-150458" alt="Pandas">
      <img src="https://img.shields.io/badge/Scikit--Learn-F7931E" alt="Scikit-learn">
    </td>
  </tr>
  <tr>
    <td><strong>Financial / Data Tools</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Dollar-Bars-4B0082" alt="Dollar-Bars">
      <img src="https://img.shields.io/badge/Triple-Barrier-FF4500" alt="Triple-Barrier Method">
      <img src="https://img.shields.io/badge/Matplotlib-008080" alt="Matplotlib">
      <img src="https://img.shields.io/badge/Seaborn-FF69B4" alt="Seaborn">
    </td>
  </tr>
  <tr>
    <td><strong>Utilities</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Jupyter-F37626" alt="Jupyter Notebook">
      <img src="https://img.shields.io/badge/Git-F05032" alt="Git">
    </td>
  </tr>
</table>

---

## Research Objectives

1. **Robust Data Structuring**: Construct an **information-dense, volatility-adjusted dataset** using Lopez de Prado’s techniques.  
2. **Advanced Temporal Modeling**: Develop a **Transformer Encoder–based architecture** to learn **multi-scale, non-linear dependencies**.  
3. **Rigorous Validation**: Apply **Purged K-Fold Cross-Validation** to avoid temporal leakage and ensure true out-of-sample generalization.  

---

# Financial Data Labeling & Validation Techniques

This section describes advanced techniques for **labeling financial time series** and **avoiding look-ahead bias during model evaluation**, based on *Advances in Financial Machine Learning* (López de Prado, 2018).

---

## Triple-Barrier Method (TBM)

The **Triple-Barrier Method (TBM)** is a robust labeling technique that generates **economically meaningful labels** for supervised learning in finance.

### What is TBM?

TBM sets **three barriers** around the entry price of a trade:

1. **Upper Barrier (Profit-Taking)** – label `+1` if price reaches a volatility-adjusted profit threshold.  
2. **Lower Barrier (Stop-Loss)** – label `-1` if price drops below a volatility-adjusted loss threshold.  
3. **Vertical Barrier (Time Limit)** – label `0` if a pre-defined time horizon expires without hitting upper/lower barriers.

### Why Use TBM?

- Produces **labels reflecting actual profit/loss events**.  
- **Adapts to market volatility** using dynamic thresholds.  
- Prevents **stale trades** with the vertical barrier.  
- Reduces **look-ahead bias**, ensuring realistic model evaluation.

### How to Implement TBM

1. **Compute volatility-adjusted thresholds**:

`Upper Barrier = p_t + k * sigma_t`  
`Lower Barrier = p_t - k * sigma_t`

Where:  
- `p_t` = price at entry  
- `sigma_t` = volatility estimate  
- `k` = threshold multiplier


2. **Monitor price until a barrier is hit** or the vertical barrier (time limit) is reached.  

3. **Assign labels**:

- `+1` → Upper barrier hit first (profit)  
- `-1` → Lower barrier hit first (loss)  
- `0` → Vertical barrier reached (neutral)

---

## Purged K-Fold Cross-Validation

**Purged K-Fold CV** is an evaluation framework designed to **avoid look-ahead bias in financial data**, which is common in traditional K-Fold CV due to **temporal dependence** between samples.

### What is Purged K-Fold?

- Data is split into `K` folds like standard K-Fold.  
- **Training samples that overlap with the test period are “purged”** to prevent leakage of future information.  
- Optionally, a **“gap”** can be introduced between training and test sets to further reduce overlap effects.

### Why Use Purged K-Fold in Finance?

- Financial time series are **non-i.i.d. and autocorrelated**; standard K-Fold can **inflate performance metrics** by leaking information.  
- Purging ensures that **no training sample contains information from the future**, producing **more realistic out-of-sample performance estimates**.  
- Particularly important when using **event-based labeling** like TBM, where price movements affect multiple sequential samples.

### Key Benefits

- **Leak-free evaluation** of predictive models  
- Accurate **out-of-sample performance estimation**  
- Reduces **false optimism** in backtesting

---

**Summary:**  

- **TBM** creates structured, economically meaningful labels for financial ML.  
- **Purged K-Fold CV** ensures **robust, leak-free model evaluation**, preventing look-ahead bias common in traditional K-Fold for time series.

---

**Reference:**  
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

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

The table below summarizes the average performance of all models across the full range of look-back and look-forward window configurations (3, 5, 7, and 10 days) on the 9-stock dataset. These results demonstrate the overall superiority of the attention-based model.

| Model | Mean Precision (Across 16 Settings) | Mean AUC Score (Across 16 Settings) |
|--------|-------------------------------------|-------------------------------------|
| **Transformer (Hybrid)** | 🟢 60.3% | 🟢 0.57 |
| Random Forest | 55.9% | 0.53 |
| Logistic Regression | 53.3% | 0.53 |
| SVM | 47.8% | 0.49 |

> The Transformer model consistently outperforms traditional methods, achieving a meaningful predictive edge with a 60.3% average precision across all experimental settings.

### 3. Insights & Discussion

- Attention layers captured **latent inter-bar dependencies**  
- LSTM hybridization enhanced **short-term recall**  
- False positives occurred during **high-volatility periods**  
- Hybrid attention + Dollar-Bars/TBM improves **robustness and interpretability**

---

## Future Directions

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

