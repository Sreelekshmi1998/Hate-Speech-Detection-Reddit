# Hate Speech Detection (NLP • ML • DL)

Classifying short social-media text into **hate / offensive / neutral**.

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

## ⚙️ Setup

```bash
# (optional) create a virtual environment
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scriptsctivate

# install dependencies
pip install -r requirements.txt
```

**Using Google Colab?**
- Mount Google Drive.
- Ensure the hardcoded paths in scripts match your Drive.
- `word_cloud.py` includes `!pip install ...` lines intentionally for Colab.

---

## 🚀 How to Run

> ⚠️ **Paths:** Some scripts use absolute Google Drive paths and `os.chdir(...)`.  
> Edit paths for local runs. In Colab, keep them as is (after mounting Drive).

### 1) Clean text

```bash
python preprocessing/clean_data.py
# reads:  hate.csv, offensive.csv, neutral.csv  (each with a 'tweet' column)
# writes: hate_clean.csv, offensive_clean.csv, neutral_clean.csv
```

### 2) Word clouds (EDA)

- **Colab recommended**: open and run `preprocessing/word_cloud.py`.
- Ensure a cleaned CSV (e.g., `hate_clean.csv`) is available at the path the script expects.

### 3) SVM baselines

```bash
# SVM (no PCA) — feature CSV with label 'lab_el'
python classical_ml/svm_no_pca.py

# SVM + PCA — feature CSV with label 'lab_el'
python classical_ml/svm_with_pca.py

# SVM with TF-IDF — raw text CSV with 'tweet', 'class'
python classical_ml/svm_tfidf.py
```

### 4) Random Forest

```bash
python classical_ml/rf_no_pca.py
python classical_ml/rf_with_pca.py
```

### 5) 1D-CNN (Keras)

```bash
python deep_learning/cnn_1d.py
```
> Some scripts save models using `joblib.dump(...)` into a Drive folder (e.g., `Voting_input/`).  
> Change or remove `os.chdir(...)` if saving locally.

---

## 📊 Suggested Results to Include (for reviewers)

Create an `assets/` folder and add:
- **Word clouds** per class: `assets/wordcloud_hate.png`, `assets/wordcloud_offensive.png`, `assets/wordcloud_neutral.png`
- **Confusion matrices**: `assets/cm_svm.png`, `assets/cm_rf.png`, `assets/cm_cnn.png`
- **CNN training curves**: `assets/cnn_training_curves.png` (accuracy/loss vs epochs)
- Optional: `assets/before_after_cleaning.png` (anonymize usernames/links)

Save a confusion matrix image (example):
```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig("assets/cm_svm.png", dpi=200, bbox_inches="tight")
```

---

## 🔎 Notes & Known Quirks

- **Paths**: Many scripts assume **Google Drive** paths; adjust for local runs.
- **Columns**: Classical models expect `lab_el`; text pipelines use `class`. Keep consistent per script.
- **Metrics scope**: Some outputs (e.g., CNN) print **train** performance. Also report test metrics for fair comparison.
- **NLTK data**: If you enable tokenizers/taggers, ensure the required NLTK resources are downloaded.

---

## 🧠 Why these methods?

- **Strong baselines**: SVM & Random Forest are robust on sparse text features (n-grams, TF-IDF) and quick to iterate.
- **1D-CNN**: Captures local word/character patterns that linear models may miss, useful for nuanced toxicity signals.
- **PCA experiments**: Tested whether dimensionality reduction helps classical models on high-dimensional feature CSVs.

---

## ⚖️ Ethics & Safety

This project involves toxic/abusive language:
- Mask or anonymize examples in screenshots.
- Do not republish raw user data without permission.
- Models can inherit dataset biases; do not deploy without careful evaluation.

---

## 📝 Disclaimer & License

This repository is shared **as portfolio evidence** of my work and is **not intended for reuse in production**.

If you want to **prevent reuse**, you can:
- Provide **no license** (default “All Rights Reserved”), or
- Use a restrictive license such as **CC BY-NC-ND 4.0** (non-commercial, no derivatives).

(If you want others to reuse, consider MIT/Apache-2.0 instead.)

---

## 🙌 Acknowledgements

This was a **group project**.  
This repository includes **only the parts I implemented**: data cleaning, word clouds, SVM, Random Forest, and 1D-CNN.

---

## 📫 Contact

If you’re reviewing this for the **Global Talent Visa** portfolio and want example runs/reports or Colab notebooks that reproduce key results, feel free to reach out.
