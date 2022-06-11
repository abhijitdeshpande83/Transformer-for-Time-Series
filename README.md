# 🚀 Exploring Deep Learning in Finance  
**Attention-Based Transformer for Robust Financial Time Series Classification**

---

## 🧩 Overview

This Master's Thesis investigates the application of **Attention-Based Transformer Encoders** to build **robust Buy/Sell classification models** for financial time series data.  

Financial markets are inherently **non-stationary, noisy, and regime-dependent**, which challenges conventional machine learning models.  
To address these issues, this research integrates **De Prado–inspired data preprocessing** with a **hybrid Transformer–LSTM architecture**, enabling improved temporal modeling and robust evaluation.

### 🔑 Key Contributions
- **De Prado–Inspired Data Pipeline**: Combines *Dollar-Bars* sampling with the *Triple-Barrier Method* for economically meaningful labeling.  
- **Hybrid Transformer–LSTM Model**: Fuses *long-range attention* with *short-term memory* for richer temporal understanding.  
- **Leak-Free Evaluation Framework**: Implements *Purged K-Fold Cross-Validation* to ensure realistic backtesting integrity.  

---

## 🎯 Research Objectives

Designing profitable quantitative models requires both **predictive accuracy** and **statistical robustness**.  
This work pursues three core objectives:

1. **Robust Data Structuring**  
   Employ *Lopez de Prado’s financial data techniques* to construct an **information-dense, volatility-adjusted dataset**.  

2. **Advanced Temporal Modeling**  
   Develop a **Transformer Encoder–based architecture** capable of learning **non-linear, multi-scale dependencies** in financial time-series data.  

3. **Rigorous Validation**  
   Apply **Purged K-Fold Cross-Validation** to eliminate temporal leakage and ensure **true out-of-sample generalization**.  

---

## 🧠 Methodology

### 1. Data Structuring Pipeline — *De Prado Framework*

#### 📊 Sampling Strategy: Dollar-Bars

| Aspect | Time Bars | Dollar-Bars *(Used Here)* |
|--------|------------|---------------------------|
| Sampling Basis | Fixed time intervals (e.g., 1h) | Fixed cumulative traded dollar volume |
| Data Quality | May contain redundant/noisy info | Equalizes economic information per bar |
| Benefit | — | Improves signal-to-noise ratio & reduces autocorrelation |

#### 🏷️ Labeling Mechanism: Triple-Barrier Method (TBM)

| Barrier | Definition | Purpose |
|----------|-------------|----------|
| **Upper Barrier** | Price rises by volatility-adjusted threshold | Profit-taking event |
| **Lower Barrier** | Price falls by volatility-adjusted threshold | Stop-loss trigger |
| **Vertical Barrier** | Time constraint | Prevents stale or unclosed trades |

#### ⚙️ Feature Engineering & Validation
- Engineered **technical indicators**: RSI, MACD, Bollinger Bands, volatility measures  
- Applied **PCA** for dimensionality reduction  
- Implemented **Purged K-Fold Cross-Validation** to prevent temporal overlap  

---

### 2. Hybrid Transformer–LSTM Architecture

| Component | Function |
|------------|-----------|
| **Positional Encoding** | Injects sequential order into input embeddings |
| **Multi-Head Self-Attention** | Learns contextual dependencies across time steps |
| **Feed-Forward Network (FFN)** | Enhances representational capacity via non-linear transformations |
| **LSTM Layer (Hybrid)** | Models localized short-term dependencies complementing global attention |

---

## 📊 Results and Performance Analysis

### 1. Experimental Setup

| Parameter | Description |
|------------|-------------|
| **Dataset** | Multi-asset OHLCV data converted to Dollar-Bars |
| **Feature Set** | Technical indicators (RSI, MACD, Bollinger Bands), PCA-reduced |
| **Labeling** | Triple-Barrier Method |
| **Validation** | Purged K-Fold (5 folds) |
| **Framework** | PyTorch |

---

### 2. Model Comparison

| Model | Mean Balanced Accuracy | Mean AUC Score |
|--------|------------------------|----------------|
| **Transformer–LSTM (Proposed)** | 🟢 **62.1%** | 🟢 **0.65** |
| Random Forest | 55.4% | 0.58 |
| Support Vector Machine (SVM) | 52.8% | 0.53 |
| Logistic Regression | 50.1% | 0.50 |

---

### 3. Insights & Discussion

#### 🧠 Learning Dynamics
- Attention layers captured **latent inter-bar dependencies** under volatile regimes.  
- The LSTM hybridization enhanced **short-term predictive recall**, balancing the model’s temporal hierarchy.  
- Dropout and normalization were crucial for **overfitting control** in low-sample regimes.  

#### 🔍 Error & Regime Analysis
- False positives aligned with **high-volatility (VIX spike)** periods.  
- Model performance declined under **low-liquidity or range-bound** market regimes.  
- Future enhancement: integrate **regime-switching volatility models** (e.g., GARCH).  

#### 💡 Key Takeaways
- **Transformers outperform traditional ML** in modeling non-stationary financial data.  
- **Dollar-Bars + TBM** provided cleaner, economically relevant event labeling.  
- **Hybrid attention architectures** enhance resilience to market shifts.  

---

### 4. Visual Results

Visualizations available under `figures/` directory:
- `feature_importance.png` — PCA component contribution  
- `roc_comparison.png` — Model ROC-AUC comparison  
- `training_curves.png` — Loss convergence trends  

---

## 🔭 Future Directions

- Explore **Temporal Fusion Transformers (TFT)** for interpretable multi-horizon forecasting.  
- Integrate **volatility clustering** features (e.g., realized volatility, GARCH).  
- Experiment with **adaptive fine-tuning** for regime-aware model updates.  

---

## 🧩 Conclusion

This research demonstrates that **attention-driven architectures** can significantly enhance robustness and interpretability in financial time-series modeling.  
By combining **data integrity (Dollar-Bars, TBM)** with **Transformer-based learning**, the model effectively captures both **macro-level structures** and **micro-level patterns**, outperforming traditional baselines in both predictive power and stability.

---

📎 **Back to Main:** [README.md](./README.md)  
📂 **Figures & Results:** [`figures/`](./figures)  
📚 **Thesis Report:** [`thesis_document.pdf`](./docs/thesis_document.pdf)
