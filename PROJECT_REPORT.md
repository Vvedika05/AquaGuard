# AquaGuard — Project Report

## 1. Introduction

AquaGuard is a supervised machine-learning system for binary classification of water-quality samples. The application receives measured water-quality parameters and produces a model prediction of Potable or Non-Potable.

The project demonstrates the full lifecycle from raw labelled data to an interactive Streamlit application.

## 2. Problem Statement

Water-quality conditions can vary across multiple physical and chemical measurements. A classification model can be used to learn patterns from historical labelled samples and provide a consistent prediction for a new set of measurements.

## 3. Objectives

1. Understand the supplied dataset.
2. Perform data-quality analysis and cleaning.
3. Conduct meaningful EDA.
4. Analyse PCA and explained variance.
5. Build multiple classification models.
6. Compare models using cross-validation.
7. Select a model using a predefined metric.
8. Persist the complete preprocessing and classifier pipeline.
9. Implement new-sample prediction.
10. Implement a live monitoring simulation.

## 4. Real-World Significance

Water quality is an important environmental monitoring problem. An interactive classification system can demonstrate how measurements can be transformed into a model prediction and monitored over time.

The system is intentionally presented as an academic ML demonstration rather than as a laboratory or regulatory water-safety instrument.

## 5. Dataset Description

The uploaded file was inspected directly before the project was built.

- Dataset shape: **3,276 × 10**
- Target column: **Potability**
- Target mapping: **0 = Non-Potable, 1 = Potable**
- Duplicate rows: **0**
- Training samples: **2,620**
- Held-out test samples: **656**

Input features:

- `ph`
- `Hardness`
- `Solids`
- `Chloramines`
- `Sulfate`
- `Conductivity`
- `Organic_carbon`
- `Trihalomethanes`
- `Turbidity`

Class distribution:

- Non-Potable: **1,998**
- Potable: **1,278**

## 6. Data Preprocessing

Missing values were found in `ph`, `Sulfate`, and `Trihalomethanes`.

No rows were removed for missingness. Numeric missing values are handled by median imputation inside the scikit-learn preprocessing pipeline. This keeps the transformation reproducible and ensures the medians are learned from training data only.

No duplicate rows were found, so duplicate removal did not change the dataset.

IQR-based outlier analysis was performed. Extreme observations were retained because blanket removal could remove legitimate observations.

The dataset was checked for negative numeric measurements and pH values outside the physical 0–14 scale. No such impossible values were found.

## 7. Exploratory Data Analysis

The EDA stage generated:

- target distribution
- missing-value analysis
- feature distributions
- boxplots
- correlation heatmap
- feature distributions by class
- selected feature-target relationships
- pairwise relationships

The analysis shows that feature scales and distributions differ considerably. The classes also show overlapping distributions across many measurements, which supports treating the task as a non-trivial classification problem rather than a deterministic thresholding exercise.

Correlation results are descriptive associations only and are not interpreted as causal relationships.

## 8. Feature Engineering

No additional domain-derived features were added.

The original measurements were retained because they directly represent the variables supplied by the dataset. Creating arbitrary ratios or interactions without a strong justification could add complexity without improving the demonstration.

Scaling is considered part of preprocessing rather than a new domain feature.

## 9. PCA / Dimensionality Reduction

PCA was applied to imputed and standardized numeric training data.

Results:

- Numeric features: **9**
- Components for ≥80% cumulative variance: **7**
- Components for ≥90% cumulative variance: **8**

PCA was used for analysis and visualization. It was not forced into the final production pipeline because the production model can operate directly on the original measured features and the project benefits from retaining feature interpretability.

## 10. Classification Algorithms

Six classifiers were compared:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine
- XGBoost

All model comparisons use the same stratified training split and reusable preprocessing.

## 11. Experimental Setup

- Train/test split: **80/20**
- Stratification: **Yes**
- Random state: **42**
- Model comparison: **5-fold StratifiedKFold**
- Test set: held out from model selection
- Preprocessing: fitted only within training folds/pipeline

The final selected pipeline is retrained on the complete training split after model selection and then evaluated once on the held-out test set.

## 12. Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Non-Potable recall
- Confusion matrix

F1-score is the primary selection metric because the observed class distribution is not perfectly balanced and F1 balances precision and recall.

Non-Potable recall is additionally reported because failing to identify a non-potable sample is an important consideration for this demonstration.

## 13. Model Comparison

Cross-validation results are stored in `outputs/model_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Non-Potable Recall |
|---|---:|---:|---:|---:|---:|---:|
| Support Vector Machine | 0.667 | 0.573 | 0.569 | 0.571 | 0.703 | 0.729 |\n| XGBoost | 0.661 | 0.606 | 0.379 | 0.466 | 0.671 | 0.842 |\n| Random Forest | 0.674 | 0.648 | 0.362 | 0.465 | 0.695 | 0.874 |\n| K-Nearest Neighbors | 0.641 | 0.583 | 0.278 | 0.376 | 0.654 | 0.873 |\n| Decision Tree | 0.626 | 0.540 | 0.270 | 0.360 | 0.593 | 0.853 |\n| Logistic Regression | 0.610 | 0.000 | 0.000 | 0.000 | 0.476 | 1.000 |\n
## 14. Best Model Selection

The selected model is **Support Vector Machine**.

Selection rule:

> Select the model with the highest 5-fold cross-validated F1-score on the training split.

The selected model achieved a cross-validated F1-score of **0.571**.

The held-out test set was not used to choose the model.

## 15. Real-Time Application

The Streamlit application loads `models/best_model.pkl`.

That artifact contains the preprocessing and classifier in one scikit-learn pipeline. Therefore, the same preprocessing used during training is applied to new user input.

The application does not retrain models.

## 16. Live Monitoring Simulation

The Live Monitoring page creates a baseline from training-data medians. The readings can change gradually during monitoring.

The **Simulate Contamination** control modifies measurements using data-derived percentile ranges. The modified sample is passed through the same persisted model.

The application compares the prediction before and after the change.

If the model prediction does not change, the application reports that the classification remained unchanged rather than fabricating an alert.

## 17. Results

Held-out test results for the selected model:

| Metric | Result |
|---|---:|
| Accuracy | 0.622 |
| Precision | 0.516 |
| Recall | 0.508 |
| F1-score | 0.512 |
| ROC-AUC | 0.644 |
| Non-Potable Recall | 0.695 |

Per-class test performance is saved in `outputs/evaluation/classification_report.csv`.

## 18. Limitations

1. The model is limited to the patterns present in the supplied dataset.
2. The held-out test performance is moderate and should not be interpreted as laboratory-grade water-quality certification.
3. The dataset contains missing values, which are imputed.
4. The monitoring page is a simulation rather than a live sensor connection.
5. The model may not generalize to different populations, geographic sources, measurement instruments, or laboratory protocols.
6. Probability outputs are model probabilities and should not be interpreted as physical certainty.

## 19. Future Scope

- Integrate IoT sensor streams.
- Add independent external validation.
- Calibrate probabilities.
- Investigate threshold tuning with domain experts.
- Add model drift monitoring.
- Add sensor anomaly detection.
- Store time-series monitoring data.
- Develop an API layer for deployment.
- Add authenticated monitoring and audit logs.

## 20. Conclusion

AquaGuard demonstrates a complete classification-based machine-learning workflow. The supplied water-quality dataset was inspected, cleaned through reproducible pipeline operations, analysed with EDA and PCA, and used to compare six classification algorithms.

The selected pipeline is persisted and integrated into Streamlit so that both new user samples and simulated live readings are classified dynamically. The application therefore demonstrates the required connection between a real-world problem, classification, model evaluation, persistence, and real-time interaction.

### Academic Disclaimer

AquaGuard is an academic machine-learning demonstration. Predictions are based on patterns learned from the supplied dataset and should not replace laboratory water-quality testing or regulatory assessment.
