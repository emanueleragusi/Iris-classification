# Iris Classification

An end-to-end machine learning walkthrough on the classic [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset), built with scikit-learn.

I am a biomedical engineer who got curious about ML and AI, and this is my first GitHub project. I use it to learn the fundamentals step by step and I document my reasoning alongside the code.

## What's inside

| File | Description |
|---|---|
| `main.py` | Full pipeline: data exploration, preprocessing, hyperparameter tuning, model comparison, cross-validation |
| `cv_results.csv` | Cross-validation results exported by the script |

## Pipeline

1. **Dataset basics:** structure, summary statistics, class balance, missing values
2. **Data exploration:** histograms, boxplots, violin plots
3. **Distribution analysis:** scatter matrix, pairplot, correlation heatmap, 3D scatter plot
4. **Relationship analysis:** pairwise feature scatter plots by species
5. **Preprocessing:** stratified 80/20 train/test split, `StandardScaler` fitted on the training set only
6. **Hyperparameter tuning:** `GridSearchCV` for Logistic Regression and SVM, `RandomizedSearchCV` for Random Forest (5-fold stratified CV)
7. **Model comparison:** `ModelComparison` class computing accuracy, precision, recall and F1 on the test set
8. **Cross-validation:** `CrossValidationAnalyzer` class (K-Fold, Stratified K-Fold, Leave-One-Out), with train vs. validation scores, overfitting gap, paired t-test between models and CSV export

## Models

Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, K-Nearest Neighbors.

## Results

5-fold stratified cross-validation on the training set (120 samples), accuracy:

| Model | CV accuracy (mean ± std) | Overfitting gap (train − validation) |
|---|---|---|
| Support Vector Machine | 0.975 ± 0.020 | −0.002 |
| Logistic Regression | 0.967 ± 0.031 | 0.015 |
| Random Forest | 0.967 ± 0.031 | 0.019 |
| K-Nearest Neighbors | 0.967 ± 0.017 | −0.004 |
| Decision Tree | 0.950 ± 0.017 | 0.050 |

All models perform similarly. The differences are within one standard deviation, so on a dataset this small they should not be over-interpreted. The Decision Tree shows the largest gap: it fits the training data perfectly (train accuracy 1.0).

Full results, including precision, recall and F1, are in [`cv_results.csv`](cv_results.csv).

## Getting started

```bash
git clone https://github.com/emanueleragusi/Iris-classification.git
cd Iris-classification
pip install pandas numpy matplotlib seaborn scipy scikit-learn
python main.py
```

The script opens several matplotlib windows and writes `cv_results.csv` in the working directory.

## Status and next steps

This is a learning project and still in progress:

- [ ] Tune the Decision Tree and KNN (currently default and `n_neighbors=3`)
- [ ] Complete the ROC/AUC analysis
- [ ] Add error analysis (confusion matrix) and feature importance
- [ ] Finish the written commentary for all guiding questions
- [ ] Add a `requirements.txt`

Feedback and suggestions are welcome. Feel free to open an issue.

## License

[Add a license, e.g. MIT]
