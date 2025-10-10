# Hate Speech Detection (NLP • ML • DL)

Classifying reddit tweets into **hate / offensive / neutral**.

This repository contains my end-to-end work on:
- **Data cleaning** for tweets
- **Exploratory word clouds** (EDA)
- **Classical ML baselines**: SVM (with/without PCA, TF-IDF) and Random Forest (with/without PCA)
- **Deep Learning**: a Keras **1D-CNN** text classifier

> **Context:** This began as a **group project**. I’m sharing **only my contributions** here as evidence for my portfolio/Global Talent application.  
> **This repo is not intended as a reusable library or production system.**

---

## ✅ My Role & Contributions

- **Preprocessing & cleaning**: HTML/URL/@mentions/RT removal, punctuation stripping, lowercasing; writes `*_clean.csv`.
- **EDA**: generated **word clouds** (with optional masking of profanity).
- **Classical ML**:
  - **SVM** (no PCA) on engineered feature CSVs
  - **SVM + PCA**
  - **SVM with TF-IDF** from raw text
  - **Random Forest** (no PCA) and **RF + PCA**
- **Deep Learning**:
  - **1D-CNN** (Keras/TensorFlow) for multiclass classification.

---

## 📁 Repository Structure

```
.
├─ preprocessing/
│  ├─ clean_data.py          # cleans "tweet" text and writes *_clean.csv
│  └─ word_cloud.py          # generates word clouds (Colab-friendly)
│
├─ classical_ml/
│  ├─ svm_no_pca.py          # SVM on feature CSVs (label column: lab_el)
│  ├─ svm_with_pca.py        # SVM + PCA on feature CSVs
│  ├─ svm_tfidf.py           # SVM on TF-IDF from raw text (tweet, class)
│  ├─ rf_no_pca.py           # Random Forest on feature CSVs
│  └─ rf_with_pca.py         # Random Forest + PCA
│
└─ deep_learning/
   └─ cnn_1d.py              # 1D-CNN on raw text (tweet, class)
```
> Scripts are preserved close to their original form. Several assume **Google Colab** + **Drive paths** (e.g., `/content/drive/...`).

---

## 🧪 Task & Labels

- Multi-class classification with labels: **`hate`**, **`offensive`**, **`neutral`**.

**Expected columns (by script type):**
- **Raw text CSVs** (for TF-IDF & CNN): `tweet`, `class`
- **Engineered feature CSVs** (for SVM/RF): many numeric feature columns + label **`lab_el`**

**Cleaning**: `preprocessing/clean_data.py` expects `hate.csv`, `offensive.csv`, `neutral.csv` (each with a `tweet` column) and writes `*_clean.csv`.  
**Stopwords**: some utilities reference `sw.txt` (space-separated words) in the working directory.

---

## 🧠 Why these methods?

- **Strong baselines**: SVM & Random Forest are robust on sparse text features (n-grams, TF-IDF) and quick to iterate.
- **1D-CNN**: Captures local word/character patterns that linear models may miss, useful for nuanced toxicity signals.
- **PCA experiments**: Tested whether dimensionality reduction helps classical models on high-dimensional feature CSVs.


---

## 📝 Disclaimer & License

This repository is shared **as portfolio evidence** of my work and is **not intended for reuse in production**.

---

## 🙌 Acknowledgements

This was a **group project**.  
This repository includes **only the parts I implemented**: data cleaning, word clouds, SVM, Random Forest, and 1D-CNN.

---

