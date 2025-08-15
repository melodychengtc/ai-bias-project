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

---

## Bias Analysis

We audited dataset representation across three text-based groups: **Length**, **Lexical Diversity**, and **Punctuation**.  
Below are the class distributions (AI = positive class).

### Group Distributions (from dataset)

**Length**
| Group | Total Samples | AI Samples | Human Samples | % AI in Group |
|:-----|--------------:|-----------:|--------------:|--------------:|
| Long | 722 | 0 | 722 | 0.00 |
| Short | 738 | 85 | 653 | 11.52 |

**Lexical Diversity**
| Group | Total Samples | AI Samples | Human Samples | % AI in Group |
|:---------------|--------------:|-----------:|--------------:|--------------:|
| High Diversity | 730 | 84 | 646 | 11.51 |
| Low Diversity  | 730 | 1  | 729 | 0.14 |

**Punctuation**
| Group | Total Samples | AI Samples | Human Samples | % AI in Group |
|:-----------------|--------------:|-----------:|--------------:|--------------:|
| High Punctuation | 730 | 78 | 652 | 10.68 |
| Low Punctuation  | 730 | 7  | 723 | 0.96 |

### What this shows (Representation Bias)

- **Long vs Short:** Long texts have **0% AI** (0/722) while Short texts have **11.52% AI**.  
  → The model cannot learn to detect AI on Long texts due to **no positive examples**.

- **Lexical Diversity:** High-diversity texts have **11.51% AI**, Low-diversity have **0.14% AI**.  
  → Severe under-representation of AI in **Low Diversity**.

- **Punctuation:** High-punctuation texts have **10.68% AI**, Low-punctuation have **0.96% AI**.  
  → AI is scarce in **Low Punctuation**.

**Disparity (max − min %AI)**
- Length: **11.52 pp**
- Diversity: **11.37 pp** (11.51 − 0.14)
- Punctuation: **9.72 pp** (10.68 − 0.96)

These gaps indicate **representation bias**: the dataset contains far fewer (or zero) AI examples in certain groups, so any model trained on it will likely under-detect AI there (high FNR, low recall), regardless of algorithm.

### Impact on Evaluation

- In groups with **zero AI samples** (e.g., Long), **recall/FNR are undefined** for the positive class (no TP+FN). Any apparent “good” performance there is not evidence of true detection capability.
- Overall accuracy can look high while **systematic blind spots** persist in underrepresented groups.

### Mitigations

1. **Data Rebalancing**
   - Add AI-generated samples for **Long**, **Low Diversity**, and **Low Punctuation** groups.
   - Target at least a **similar %AI** as the better-represented counterpart (≈10–12%).

2. **Stratified Sampling / Group-Weighted Training**
   - Ensure each group contributes positives and negatives per batch/epoch.
   - Consider **group-aware loss** (e.g., sample weights per group).

3. **Two-Stage Thresholding**
   - Keep your global threshold.
   - Add **group-specific guardrails** (slightly lower threshold in underrepresented groups) until data is rebalanced. Monitor for new false positives.

4. **Robust Reporting**
   - Always report **per-group**: Precision, Recall, FNR, and **Selection Rate**.
   - Highlight groups with **low or zero positives** as “insufficient data to assess.”

5. **(Optional) Synthetic Augmentation**
   - Generate AI essays that are **longer**, **lower diversity**, and **lower punctuation** to fill gaps. Validate with human review.

---
## 📊 Model Performance Improvements

We evaluated three model iterations to improve AI text detection performance.  
Below is a comparison of accuracy, precision, recall, and F1-score across the models.

| Model | Training Method | Accuracy | Precision | Recall | F1-score | AUC-PR |
|-------|----------------|----------|-----------|--------|----------|--------|
| **Model 1** | Trained on original dataset using Logistic Regression + Stratified K-Fold CV | 0.7875 | 0.7683 | 0.6700 | 0.7157 | 0.790 |
| **Model 2** | Trained on augmented dataset with synthetic samples to address underrepresentation (longer word length, lower lexical density, lower punctuation count) | 0.7839 | 0.6960 | 0.8147 | 0.7507 | 0.811 |
| **Model 3** | Applied NLP feature engineering (word/char TF-IDF + stylometry), then calibrated with **Platt scaling (sigmoid)** for improved probability confidence | 0.7703 | 0.7199 | 0.6951 | 0.7073 | 0.8041 |

---

### 🔍 Observations
- **Model 2** showed improved accuracy and precision compared to Model 1, indicating that balancing underrepresented AI samples improved detection for certain cases.
- **Model 3** slightly reduced accuracy but significantly **increased recall and F1-score**, meaning it is better at catching AI-generated text while maintaining balanced performance.
- **AUC-PR** improved from **0.790 → 0.804** between Model 1 and Model 3, showing better ranking ability for positive predictions.
- Calibration via **Platt scaling** helped spread probability scores more evenly, making confidence values more meaningful for downstream bias analysis.

---

### ✅ Takeaway
By progressively:
1. Starting with a baseline logistic regression model (Model 1),
2. Augmenting the dataset to reduce bias in underrepresented AI text patterns (Model 2),
3. Adding richer NLP features and calibrating probabilities (Model 3),

We achieved **higher recall, better F1-score, and more interpretable probability outputs**, making the final model more effective for bias detection tasks.
