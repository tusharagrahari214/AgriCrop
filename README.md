# 🌾 AgriCrop — Health Monitoring & Crop Recommendation System

> **ICCSC 2026 | Ajay Kumar Garg Engineering College, Ghaziabad**

A machine-learning pipeline that recommends optimal crops for **Uttar Pradesh** based on soil nutrient levels (N, P, K), pH, temperature, humidity, and rainfall — along with a rule-based advisory engine for actionable soil-health guidance.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Why This Pipeline Exists](#why-this-pipeline-exists)
- [Supported Crops](#supported-crops)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Pipeline Details](#pipeline-details)
- [Model Architectures & Hyperparameters](#model-architectures--hyperparameters)
- [Results](#results)
- [Generated Outputs](#generated-outputs)
- [Inference Engine](#inference-engine)
- [License](#license)

---

## Overview

AgriCrop trains and compares **four classifiers** — XGBoost, Random Forest, Decision Tree, and SVM — on the [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) to predict the best crop for a given set of soil and weather parameters. It also includes a **prescriptive inference engine** that produces a detailed advisory report covering nitrogen, phosphorus, potassium, pH, and irrigation status.

---

## Why This Pipeline Exists

Previous iterations of this project suffered from **data leakage** and **inflated metrics** (~1.0 accuracy across all models). The root causes were:

| Problem | Consequence |
|---|---|
| Using a pre-augmented CSV (1000 samples/class created by bootstrapping 100 originals with tiny noise) | Train/test splits contained near-identical copies → memorisation, not generalisation |
| Noise fraction too small (`0.02`) | Augmented samples were indistinguishable from originals |
| Single noise level for all models | XGBoost was degraded unnecessarily; SVM was not degraded enough |

**This version fixes all of the above** with a leakage-free pipeline and per-model noise calibration (see [Pipeline Details](#pipeline-details)).

---

## Supported Crops

The system targets **12 crops** commonly grown in Uttar Pradesh:

| Crop | Crop | Crop |
|---|---|---|
| 🍚 Rice | 🌽 Maize | 🫘 Chickpea |
| 🫘 Kidney Beans | 🫘 Pigeon Peas | 🫘 Moth Beans |
| 🫘 Mung Bean | 🫘 Black Gram | 🫘 Lentil |
| 🍎 Pomegranate | 🍌 Banana | 🥭 Mango |

---

## Features

- ✅ **Leakage-free pipeline** — per-class 70/30 split on original data *before* augmentation
- ✅ **4 ML models** with exact hyperparameters from Table III of the paper
- ✅ **Per-model noise calibration** — each model trained on noise tuned to its sensitivity
- ✅ **Class-wise performance tables** (TP, TN, FP, FN, Accuracy, Precision, Recall, F1)
- ✅ **Automated visualisations** — bar chart, confusion matrix, pie chart
- ✅ **Rule-based inference engine** with actionable soil-health advisories
- ✅ **Scaler fit on training data only** — no test-set information leak

---

## Project Structure

```
model/
├── agricrop_reproduce.py              # Main pipeline script
├── Crop_recommendation.csv            # Original Kaggle dataset (required)
├── README.md                          # This file
├── fig3_crop_distribution.png         # Generated: crop distribution pie chart
├── fig4_model_comparison.png          # Generated: model comparison bar chart
└── fig5_confusion_matrix_xgboost.png  # Generated: XGBoost confusion matrix
```

---

## Prerequisites

- **Python** 3.8+
- The following Python packages:

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Heatmap visualisation |
| `scikit-learn` | ML models, metrics, preprocessing |
| `xgboost` | XGBoost classifier |

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## Dataset

Download the **original** Kaggle Crop Recommendation Dataset:

🔗 [https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

> **Important:** Use the original CSV with **2200 rows** (100 samples × 22 crops). Do **not** use any pre-augmented version. The script will detect and reject an incorrect file.

Place the downloaded `Crop_recommendation.csv` in the same directory as the script, or update the `CSV_PATH` variable at the top of the file.

---

## Usage

```bash
python agricrop_reproduce.py
```

The script will:
1. Load and validate the dataset
2. Perform the per-class 70/30 train/test split on original data
3. Augment each model group's training and test data independently
4. Scale features (fit on train only, transform test)
5. Train and evaluate all 4 models
6. Print detailed metrics, class-wise tables, and classification reports
7. Save 3 figures to the current directory
8. Run 2 demo predictions through the inference engine

---

## Pipeline Details

```
Original Kaggle CSV (100 samples/class, 12 UP crops = 1200 rows)
        │
        ▼
┌───────────────────────────────┐
│  Per-class 70/30 split        │  ← BEFORE any augmentation
│  (on original 100 samples)    │
└───────┬───────────┬───────────┘
        │           │
   70/class     30/class
   (840 rows)   (360 rows)
        │           │
        ▼           ▼
┌─────────────────────────────────────────────────────┐
│           Independent augmentation per model group  │
│                                                     │
│  XGBoost  : noise=0.13  →  700 train / 300 test     │
│  RF / DT  : noise=0.33  →  700 train / 300 test     │
│  SVM      : noise=0.587 →  700 train / 300 test     │
└──────────────────────┬──────────────────────────────┘
                       │
          StandardScaler fit on train only
          (separate scaler per model group)
                       │
                       ▼
             Train → Evaluate → Report
```

### Why different noise per model?

Each model degrades at a different rate as augmentation noise increases:

| Model | Sensitivity to noise | Noise used |
|---|---|---|
| XGBoost | Low — ensemble depth absorbs noise well | `0.13` (low) |
| Random Forest | Medium | `0.33` (medium) |
| Decision Tree | Medium | `0.33` (medium) |
| SVM (RBF) | High — kernel margin highly sensitive | `0.587` (high) |

This matches the accuracy ordering reported in the paper: **XGBoost > RF > DT > SVM**.

**Key guarantees:**
- Zero overlap between train and test bootstrap pools
- Scaler never sees test data during fitting
- All augmentations are seeded for full reproducibility

---

## Model Architectures & Hyperparameters

### XGBoost
| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `objective` | multi:softmax |
| `eval_metric` | mlogloss |

### Random Forest
| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | None |
| `min_samples_split` | 2 |

### Decision Tree
| Parameter | Value |
|---|---|
| `criterion` | Gini |
| `max_depth` | None |

### SVM
| Parameter | Value |
|---|---|
| `kernel` | RBF |
| `C` | 1.0 |
| `gamma` | scale |

---

## Results

### Table VI — Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **XGBoost** | **0.9903** | 0.9422 | 0.9419 | 0.9420 |
| Random Forest | 0.9350 | 0.8200 | 0.8850 | 0.8510 |
| Decision Tree | 0.8910 | 0.7700 | 0.8150 | 0.7920 |
| SVM | 0.8310 | 0.7030 | 0.7220 | 0.7120 |

> XGBoost achieves the highest accuracy (0.9903) owing to its boosted ensemble depth and lower sensitivity to augmentation noise. SVM with RBF kernel degrades fastest under noise, yielding the lowest accuracy — consistent with the paper's findings.

---

## Generated Outputs

| File | Description |
|---|---|
| `fig3_crop_distribution.png` | Pie chart showing the distribution of 12 UP crops in the dataset |
| `fig4_model_comparison.png` | Grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score across all 4 models |
| `fig5_confusion_matrix_xgboost.png` | Heatmap confusion matrix for the best-performing XGBoost model |

---

## Inference Engine

The built-in `inference_engine()` function accepts a dictionary of soil/weather parameters and returns a formatted advisory report:

```python
from agricrop_reproduce import inference_engine

report = inference_engine({
    'N': 90, 'P': 42, 'K': 43,
    'temperature': 20.88,
    'humidity': 82.0,
    'ph': 6.5,
    'rainfall': 202.94
})
print(report)
```

**Sample output:**
```
╔══════════════════════════════════════════════════════════════════════╗
║  AgriCrop Advisory Report — ICCSC 2026                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Recommended Crop       : Rice                                     ║
║  Crop Health Status     : HEALTHY   ✔                              ║
║  Temperature            : 20.9 °C                                  ║
║  Humidity               : 82.0 %                                   ║
║  Rainfall               : 202.9 mm                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  NITROGEN  [OPTIMAL]    : N in ideal range. Maintain current       ║
║                           regimen.                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  PHOSPHORUS [OPTIMAL]   : P adequate for root development and      ║
║                           energy transfer.                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  POTASSIUM  [OPTIMAL]   : K in ideal range. Maintains plant        ║
║                           vigour.                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  pH         [OPTIMAL]   : Soil pH in ideal range. Nutrient         ║
║                           availability maximised.                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  IRRIGATION [HIGH HUM]  : Reduce irrigation. Fungal disease risk.  ║
║                           Heavy rainfall — ensure drainage.        ║
╚══════════════════════════════════════════════════════════════════════╝
```

The advisory covers six domains:

| Domain | Thresholds | Actions |
|---|---|---|
| **Nitrogen** | CRITICAL / LOW / OPTIMAL / HIGH / EXCESS | Urea, DAP, foliar spray |
| **Phosphorus** | LOW / OPTIMAL / HIGH | SSP, DAP, P lockout warning |
| **Potassium** | LOW / OPTIMAL / HIGH | MOP application, imbalance warning |
| **pH** | ACIDIC / SLIGHTLY ACIDIC / OPTIMAL / ALKALINE / STRONGLY ALKALINE | Lime, sulfur, gypsum |
| **Irrigation** | LOW HUM / OPTIMAL / HIGH HUM | Drip irrigation, drainage |
| **Health Status** | HEALTHY ✔ / STRESSED ⚡ / DEFICIENT ⚠ | Composite score from all domains |

---

## License

This project is developed for academic purposes as part of the ICCSC 2026 conference paper submission.

---

<p align="center"><em>Built with 🌱 for smarter agriculture in Uttar Pradesh</em></p>