# AI-Generated Text Detection

A machine learning pipeline to detect AI-generated essays from the [AI-Generated Essays Dataset](https://www.kaggle.com/datasets/denvermagtibay/ai-generated-essays-dataset).  
Built with **Python**, **scikit-learn**, and **pandas** — optimized for **high recall** while reducing false positives.

---

## 📌 Project Overview

This project trains a classification model to distinguish AI-generated from human-written essays.

**Key goals:**
- Catch as many AI essays as possible (**high recall**)
- Reduce wrongful human flags (**minimize false positives**)
- Maintain strong overall accuracy

---

## 🛠 Workflow

### 1. Dataset Acquisition
- Downloaded using [KaggleHub](https://github.com/Kaggle/kagglehub)
- Loaded into Pandas for preprocessing

### 2. Feature Engineering
Extracted linguistic and statistical features:
- Word count
- Sentence length
- Lexical diversity
- Punctuation frequency

Labels:  
`1 = AI-generated`  
`0 = Human`

### 3. Model Pipeline
- **StandardScaler** → normalize feature values
- **Logistic Regression** (`class_weight="balanced"`)
- **Probability Calibration** (`isotonic`) for better threshold tuning

### 4. Evaluation
- Metrics: **Accuracy**, **Precision**, **Recall**, **F1-score**
- **Stratified K-Fold Cross-Validation** for stability checks

### 5. Threshold Optimization
- Swept probability thresholds from `0 → 1`
- Selected threshold where:
  - Recall remains ≥ 0.98
  - Precision maximized
- Evaluated with a **confusion matrix** to track:
  - **TP**: AI correctly flagged
  - **TN**: Humans correctly passed
  - **FP**: Humans wrongly flagged
  - **FN**: AI missed

---

## 📊 Results

| Metric          | Original Model | Fine-Tuned Threshold |
|-----------------|----------------|----------------------|
| Accuracy        | 0.984          | 0.987                |
| Precision       | 0.789          | 0.865                |
| Recall          | 1.000          | 0.985                |
| F1-score        | 0.882          | 0.922                |
| False Positives | Higher         | Reduced              |
| False Negatives | 0              | Slight increase      |

**Summary:**
- **Original model**: Perfect recall but moderate precision → more human texts flagged incorrectly.
- **Fine-tuned model**: Slight recall drop, **big precision gain**, fewer false positives, and improved overall accuracy.
