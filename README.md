# 📝 AI-Generated Text Detection with Bias-Aware Training

A machine learning pipeline to detect AI-generated essays from the [AI-Generated Essays Dataset](https://www.kaggle.com/datasets/denvermagtibay/ai-generated-essays-dataset).  
Built with **Python**, **scikit-learn**, and **pandas** — optimized for **high recall** while reducing false positives and mitigating representation bias.

---

## 📌 Project Overview

This project trains a classification model to distinguish AI-generated from human-written essays.

**Key goals:**
- Catch as many AI essays as possible (**high recall**)
- Reduce wrongful human flags (**minimize false positives**)
- Maintain strong overall accuracy
- Reduce dataset representation bias across linguistic subgroups

---

## 🛠 Workflow

### 1. Dataset Acquisition
- Downloaded using [KaggleHub](https://github.com/Kaggle/kagglehub)
- Loaded into Pandas for preprocessing

### 2. Bias Audit
Before training, we audited the dataset for representation bias across:
- **Length** (Long / Short essays)
- **Lexical Diversity**
- **Punctuation Frequency**

We found significant underrepresentation of AI essays in:
- **Long** essays (0% AI)
- **Low Diversity** essays (0.14% AI)
- **Low Punctuation** essays (0.96% AI)

### 3. Feature Engineering
Extracted linguistic and statistical features:
- Word count
- Sentence length
- Lexical diversity
- Punctuation frequency
- TF-IDF (word & char)
- Stylometry features

Labels:  
`1 = AI-generated`  
`0 = Human`

### 4. Model Pipeline
- **TF-IDF + Stylometry** feature union
- **Logistic Regression** (`class_weight="balanced"`) or **Linear SVC**
- **Probability Calibration** (Platt scaling / isotonic) for better threshold tuning
- **Stratified K-Fold Cross-Validation** for stability checks

### 5. Threshold Optimization
- Swept probability thresholds from `0 → 1`
- Selected thresholds maximizing either **F1** or **AUC-PR** depending on use case
- Evaluated per-group metrics to monitor bias

---

## 📊 Representation Bias Analysis

| Group | Total Samples | AI Samples | Human Samples | % AI in Group |
|-------|--------------:|-----------:|--------------:|--------------:|
| **Length – Long** | 722 | 0 | 722 | 0.00 |
| **Length – Short** | 738 | 85 | 653 | 11.52 |
| **Lexical – High** | 730 | 84 | 646 | 11.51 |
| **Lexical – Low**  | 730 | 1 | 729 | 0.14 |
| **Punctuation – High** | 730 | 78 | 652 | 10.68 |
| **Punctuation – Low**  | 730 | 7  | 723 | 0.96 |

**Disparity (max − min %AI)**  
- Length: **11.52 pp**  
- Diversity: **11.37 pp**  
- Punctuation: **9.72 pp**

These imbalances cause blind spots — the model struggles to detect AI in groups with near-zero AI samples.

---

## 🧪 Model Iterations & Results

| Model | Training Method | Accuracy | Precision | Recall | F1-score | AUC-PR |
|-------|----------------|----------|-----------|--------|----------|--------|
| **Model 1** | Original dataset, Logistic Regression + Stratified K-Fold CV | 0.7875 | 0.7683 | 0.6700 | 0.7157 | 0.790 |
| **Model 2** | Augmented dataset with synthetic AI essays for underrepresented groups | 0.7839 | 0.6960 | 0.8147 | 0.7507 | 0.811 |
| **Model 3** | TF-IDF + Stylometry features, Platt scaling for calibrated probabilities | 0.7703 | 0.7199 | 0.6951 | 0.7073 | 0.860 |

---

### 🔍 Observations
- **Model 2** increased recall by adding synthetic samples in bias-heavy groups.
- **Model 3** improved **AUC-PR** substantially (0.790 → 0.860), meaning better ranking of AI vs human essays.
- Calibration improved probability spread, allowing more effective threshold tuning for different business cases.

---

## ✅ Key Takeaways
1. **Data augmentation** mitigated group-level bias.
2. **Richer NLP features** improved discrimination ability.
3. **Probability calibration** produced more reliable confidence scores.
4. Evaluating **per-group metrics** is essential to avoid hidden blind spots.

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/ai-essay-detection.git
cd ai-essay-detection
pip install -r requirements.txt
