# Iris Classification

An end-to-end machine learning walkthrough on the classic [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset), built with scikit-learn.

I am a biomedical engineer who got curious about ML and AI, and this is my first GitHub project. I use it to learn the fundamentals step by step and I document my reasoning alongside the code.

## What's inside

| File | Description |
|---|---|
| [`Iris_classification.ipynb`](Iris_classification.ipynb) | Notebook version with all outputs and plots already rendered (readable directly on GitHub) |
| `main.py` | Same pipeline as a plain Python script |
| `cv_results.csv` | Nested cross-validation results exported by the script |

## Pipeline

1. **Dataset basics:** structure, summary statistics, class balance, missing values
2. **Data exploration:** mean/median/std, IQR outlier check, histograms, boxplots, violin plots
3. **Distribution analysis:** pairplot, correlation heatmap, 3D scatter plot
4. **Relationship analysis:** pairwise feature scatter plots by species
5. **Train/test split:** stratified 80/20
6. **Hyperparameter tuning:** `GridSearchCV` / `RandomizedSearchCV` for all five models (5-fold stratified CV)
7. **Model comparison:** nested cross-validation via the `CrossValidationAnalyzer` class (accuracy, precision, recall, F1, train vs. validation gap, paired t-test, CSV export)
8. **Final evaluation:** classification report, confusion matrix and one-vs-rest ROC curves on the test set

### Methodology

- The `StandardScaler` is part of each model `Pipeline`, so it is fitted only on training folds (no data leakage).
- Models are compared with **nested cross-validation**: hyperparameters are tuned inside each outer training fold, and scores are measured on data the tuning never saw.
- The **test set is used only once**, on the model selected by cross-validation.
- Ties in mean accuracy are broken by lower standard deviation, then by lower overfitting gap.

## Models

Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN).

## Results

Nested 5-fold stratified cross-validation on the training set (120 samples), accuracy:

| Model | CV accuracy (mean ± std) | Overfitting gap (train − validation) |
|---|---|---|
| SVM | 0.967 ± 0.031 | 0.008 |
| Random Forest | 0.967 ± 0.031 | 0.010 |
| Logistic Regression | 0.958 ± 0.026 | 0.019 |
| KNN | 0.958 ± 0.026 | 0.025 |
| Decision Tree | 0.942 ± 0.020 | 0.042 |

All models perform similarly: the differences are within one standard deviation, so on a dataset this small no model is clearly better. The Decision Tree has the largest gap, meaning it memorizes the training folds.

**Test set (30 samples), selected model: SVM (linear kernel, C = 0.1):** accuracy 0.933 (28/30 correct). *Setosa* is classified perfectly; the errors are between *versicolor* and *virginica*, which overlap in feature space. With only 30 test samples, a single mistake moves accuracy by about 3 points.

Full results, including precision, recall and F1, are in [`cv_results.csv`](cv_results.csv).

## Getting started

```bash
git clone https://github.com/emanueleragusi/Iris-classification.git
cd Iris-classification
pip install pandas numpy matplotlib seaborn scipy scikit-learn
python main.py
```

The script opens several matplotlib windows and writes `cv_results.csv` in the working directory. The nested cross-validation takes about two minutes. To explore the results without running anything, open the notebook.

## Status and next steps

This is a learning project and still in progress.

- [x] Tune all five models
- [x] Nested cross-validation and statistical comparison
- [x] Confusion matrix and ROC curves
- [ ] Error analysis of the misclassified samples
- [ ] Feature importance
- [ ] Repeated cross-validation to reduce the noise of a small dataset
- [ ] Add a `requirements.txt`

Feedback and suggestions are welcome. Feel free to open an issue.
