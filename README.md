# Machine Learning Lab: Polynomial and Logistic Regression

![Python](https://img.shields.io/badge/python-3.14-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange) ![Jupyter](https://img.shields.io/badge/notebook-Jupyter-informational)

| | |
|---|---|
| **Course** | Machine Learning, module *Training Models* |
| **Assignment** | Lab of 4 November 2026 |
| **Submitted to** | Prof. Giacomo Fiumara |
| **Author** | Nasywa Ramadhani, 574106 |
| **Deliverable** | [`Notebook.ipynb`](Notebook.ipynb) |

## Overview

This repository contains my completed lab notebook. It applies the techniques from the lecture to real data and covers two topics:

1. **Polynomial regression** on the scikit-learn diabetes dataset, with an analysis of how model complexity affects generalization.
2. **Logistic regression** on the Iris dataset, from the sigmoid and the log-loss cost to decision boundaries and multiclass softmax regression.

The notebook is submitted fully executed (Restart & Run All), with all code exercises completed and all written answers filled in.

## Contents

### Part 1: Polynomial regression (diabetes dataset, 442 patients, 10 features)

| Exercise | Topic |
|:-:|---|
| 1 | Linear regression on a single feature (`bmi`) |
| 2 | Polynomial fits of degree 1, 2, 5 and 10 on `bmi` |
| 3 | Polynomial regression on all 10 features (degrees 1 to 3): train vs. test error |
| 4 | Inspection of the predictions of the overfitted degree-3 model |

### Part 2: Logistic regression (Iris dataset)

| Exercise | Topic |
|:-:|---|
| 5 | Sigmoid function implemented from scratch |
| 6 | Logistic cost function (log loss) |
| 7 | Decision boundary on one feature (petal width, virginica vs. rest) |
| 8 | Effect of the classification threshold (false positives vs. false negatives) |
| 9 | Decision boundary on two features, with a train/test evaluation |
| 10 | Softmax regression on three classes, with manual verification of the probabilities |

### Part 3: Open questions

Five conceptual questions (Q1 to Q5), answered in the notebook.

## Key results

| Experiment | Result |
|---|---|
| Linear fit on `bmi` only | Test R² ≈ 0.28, test MSE ≈ 3885 |
| Polynomial regression, all features | Degree 1 is best on the test set (MSE ≈ 2822). Degree 3 builds 286 features from 309 training samples: train MSE ≈ 585, test MSE ≈ 76,700 (severe overfitting) |
| Logistic regression, petal width | Decision boundary at ≈ 1.66 cm, accuracy 0.96 |
| Threshold 0.1 / 0.5 / 0.9 | False positives 40 / 2 / 0; false negatives 0 / 4 / 33 |
| Logistic regression, two features | Train accuracy 0.955, test accuracy 1.000 (75/25 split, 38 test samples) |
| Softmax regression | Flower with petal length 5 cm and width 2 cm: virginica, P ≈ 0.971. Manual softmax matches `predict_proba` |

**Notes**

- The test accuracy in Exercise 9 comes from a small test set, so it is a noisy estimate. Cross-validation would be a more reliable evaluation.
- The degree-3 test MSE is numerically sensitive, because the design matrix is nearly singular. Its exact value may vary across library versions, but the conclusion does not change.

## Repository structure

```
.
├── README.md
├── requirements.txt
└── Notebook.ipynb
```

## Reproducing the results

No external data files are needed: both datasets are loaded through `sklearn.datasets`.

```bash
git clone <repository-url>
cd <repository-folder>
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Notebook.ipynb
```

Then select **Kernel → Restart & Run All**.

## Environment

Python 3.14, NumPy, pandas, Matplotlib, scikit-learn and Jupyter. See [`requirements.txt`](requirements.txt).
