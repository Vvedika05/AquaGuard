# AquaGuard

## Real-Time Water Safety Monitoring & Classification System

AquaGuard is an end-to-end academic machine-learning application that classifies a water-quality sample as **predicted Potable (1)** or **predicted Non-Potable (0)**. It includes data-quality analysis, EDA, PCA analysis, multiple classification models, model comparison, model persistence, new-sample prediction, and a live monitoring simulation.

> **Academic disclaimer:** AquaGuard is an academic machine-learning demonstration. Predictions are based on patterns learned from the supplied dataset and should not replace laboratory water-quality testing or regulatory assessment.

## Live Demo

**AquaGuard:** [Open the live application](https://aquaguard-ml.streamlit.app/)

## Problem Statement

Water-quality measurements can vary across physical and chemical parameters. The project demonstrates how supervised classification can learn patterns in labelled water-quality data and provide an interactive prediction for a new sample.

## Objectives

- Inspect and clean the supplied labelled dataset.
- Explore distributions, missingness, relationships, correlations, and class balance.
- Analyse PCA as a dimensionality-reduction technique.
- Compare six classification algorithms.
- Select the final model using a predefined F1-based selection rule.
- Persist preprocessing and classification together in one pipeline.
- Provide interactive prediction for new user input.
- Demonstrate a live monitoring simulation without retraining the model.

## Dataset

The uploaded dataset contains **3,276 rows and 10 columns**. The target was detected as `Potability` and contains the documented mapping **0 = Non-Potable, 1 = Potable**.

After duplicate handling, the dataset remains **3,276 × 10** because no exact duplicate rows were found.

Class distribution:
- Non-Potable (0): **1,998**
- Potable (1): **1,278**

Missing values were observed in:
- `ph`: 491
- `Sulfate`: 781
- `Trihalomethanes`: 162

## Dataset Source

The dataset used in AquaGuard was obtained from Kaggle:

**Water Potability Dataset — Aditya Kadiwal**

https://www.kaggle.com/datasets/adityakadiwal/water-potability/data

## Features

The model inputs are:
`ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity`

All supplied predictors are numeric in the uploaded dataset.

## Data Preprocessing

- Exact duplicates were checked; none were found.
- Missing numeric values are handled with **median imputation inside the training pipeline**.
- Numeric variables are standardized with `StandardScaler`.
- Categorical handling is included in the reusable `ColumnTransformer` for robustness, although the supplied dataset has no categorical predictors.
- No blanket outlier deletion was applied. IQR-based outlier counts are recorded in `outputs/eda/outlier_analysis.csv`.
- No additional domain-derived features were added because the supplied measurements already provide direct predictors and unnecessary engineered ratios could reduce interpretability or introduce assumptions.

All learned preprocessing parameters are fitted only on the training split and stored with the final model.

## EDA

EDA outputs are saved under `outputs/eda/` and include:
- class distribution
- missing-value analysis
- numerical distributions
- boxplots
- correlation heatmap
- feature distributions by class
- selected feature relationships
- pairwise relationships
- PCA explained variance
- PCA cumulative variance
- 2D PCA projection
- model comparison visualization

`outputs/eda/data_quality_summary.csv` and `outputs/eda/descriptive_statistics.csv` contain tabular summaries.

## PCA

PCA was fitted only on standardized numeric training data for analysis.

- Number of numeric features analysed: **9**
- Components explaining at least 80% cumulative variance: **7**
- Components explaining at least 90% cumulative variance: **8**

PCA is **not included in the final production pipeline.**. It is retained as an analytical dimensionality-reduction component because the production pipeline is evaluated in the original feature space and the project prioritizes interpretability and direct use of the measured parameters.

## Machine Learning Models

The following models were compared using 5-fold stratified cross-validation on the training split:

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. Support Vector Machine
6. XGBoost

## Model Evaluation

Reported metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Non-Potable recall

The final selection metric is **F1-score**. This balances precision and recall and prevents selection of a degenerate classifier that predicts only one class. The held-out test set is not used for model selection.

## Model Comparison

The cross-validated comparison is stored in `outputs/model_comparison.csv`.

The selected model is **Support Vector Machine**, with cross-validated F1 of **0.571**.

### Held-out test results

| Metric | Result |
|---|---:|
| Accuracy | 0.622 |
| Precision | 0.516 |
| Recall | 0.508 |
| F1 | 0.512 |
| ROC-AUC | 0.644 |
| Non-Potable Recall | 0.695 |

The test set contained **656** unseen samples.

## Real-Time Application

`app.py` loads `models/best_model.pkl`. The application does **not** retrain a model.

### Single Sample mode

1. Enter a new sample.
2. Review observed training ranges.
3. Click **Analyze Water**.
4. The sample is converted to a DataFrame.
5. The persisted preprocessing + classifier pipeline generates a prediction and, where supported, a potable probability.

If a value is outside the observed training range, the app warns that it is outside the training distribution. It does not automatically label such a value scientifically unsafe.

### Live Monitoring mode

The monitoring page starts from a baseline derived from training-data medians. When monitoring is active, simulated readings can change gradually and the persisted model is run again on the updated sample.

### Simulate Contamination

The button modifies selected measurements using changes derived from observed training percentiles. It then runs the persisted model again.

The application never hard-codes a desired prediction. If the prediction changes, the before/after state is shown. If it does not change, the application explicitly reports that the classification remained unchanged.

## Project Structure

```text
AquaGuard/
├── data/
│   └── water_potability.csv
├── models/
│   └── best_model.pkl
├── outputs/
│   ├── eda/
│   ├── evaluation/
│   ├── model_comparison.csv
│   ├── model_config.json
│   ├── predictions.csv
│   ├── feature_ranges.csv
│   ├── pca_summary.json
│   └── cleaning_log.json
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── train_models.py
│   ├── model_selection.py
│   ├── evaluate.py
│   └── predict.py
├── app.py
├── README.md
├── PROJECT_REPORT.md
├── requirements.txt
└── .gitignore
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train / Rebuild the Model

From the project root:

```bash
python src/train_models.py
```

This regenerates EDA, PCA, comparison, evaluation, metadata, predictions, and `models/best_model.pkl`.

## Run the Streamlit Application

```bash
streamlit run app.py
```

## How to Use

### 1. Single Sample

Open **Water Analysis**, enter a new water-quality sample, and select **Analyze Water**.

### 2. Live Monitoring

Open **Live Monitoring**:
- Start Monitoring
- Pause
- Simulate Contamination
- Restore Baseline
- Reset

The simulation calls the persisted model for the current readings; it does not display prerecorded prediction results.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit
- Joblib

## Machine Learning

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine
- XGBoost
- PCA
- Stratified 5-Fold Cross-Validation
  
## Limitations

- The model reflects patterns in the supplied dataset and may not generalize to other water sources, regions, instruments, or laboratory protocols.
- The dataset contains missing measurements and the application handles these through training-set median imputation.
- Classification performance is moderate on the held-out test set.
- A model prediction is not a laboratory measurement or regulatory determination.
- The live monitor is a software simulation, not a physical sensor connection.

## Future Scope

- Connect validated IoT water-quality sensors.
- Add timestamped streaming ingestion.
- Add drift detection and model monitoring.
- Evaluate calibrated probabilities and threshold selection with domain stakeholders.
- Validate on independent laboratory datasets.
- Add secure data storage and audit trails.
- Compare against additional tuned ensemble models.

