import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_validate, cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D

# Set the style for seaborn
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# TASK num. 1: Dataset Basics 
iris = load_iris()
dataframe = pd.DataFrame(data=iris.data, columns=iris.feature_names) # lets create a DataFrame from the iris dataset
dataframe['target'] = iris.target # add special column for target
dataframe['species'] = iris.target
dataframe['species'] = dataframe['species'].map(dict(enumerate(iris.target_names)))
print(f"Displays the first 5 rows of the DataFrame: {dataframe.head()}")
print(f"Provides summary statistics for numerical columns: {dataframe.describe()}")
print(f"Shows a summary of the DataFrame structure: {dataframe.info()}")
print(f"(Number of rows, Number of columns): {dataframe.shape}")
print(f"Missing values: {dataframe.isnull().sum()}")

# TASK num. 2: Data Exploration
fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # Create a histogram for each feature in the dataset
axes = axes.ravel()
for i, column in enumerate(iris.feature_names):
    axes[i].hist(dataframe[column], bins=20, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Histogram of {column}')
    plt.subplots_adjust(hspace=0.5)

plt.figure(figsize=(12, 8)) # Creare a boxplot for each feature in the dataset
for i, column in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=column, data=dataframe)
    plt.title(f'Boxplot of {column} by Species')
    plt.subplots_adjust(hspace=0.5)

plt.figure(figsize=(12, 8)) # Create a violin plot for each feature in the dataset
for i, column in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.violinplot(x='species', y=column, data=dataframe, hue='species', palette='pastel', legend=False)
    plt.title(f'Violin plot of {column} by Species')

plt.subplots_adjust(hspace=0.5)
plt.show()

# TASK num. 3: Distribution Analysis
pd.plotting.scatter_matrix(dataframe)
plt.suptitle("Scatter Matrix", y=1.02)
sns.pairplot(dataframe, hue='species', palette='husl')
numeric_df = dataframe.select_dtypes(include='number') # Only use numeric columns for correlation
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# 3D Scatter Plot (using any 3 features)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x = dataframe['sepal length (cm)']
y = dataframe['sepal width (cm)']
z = dataframe['petal length (cm)']
species = dataframe['species']
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for s in species.unique():
    ax.scatter(
        x[species == s], 
        y[species == s], 
        z[species == s], 
        label=s, 
        color=colors[s]
    )
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
ax.set_title('3D Scatter Plot of Iris Features')
ax.legend()
plt.show()

# TASK num. 4: Relationship Analysis
plt.figure(figsize=(8, 6)) # Sepal Length vs Sepal Width
sns.scatterplot(data=dataframe, x='sepal length (cm)', y='sepal width (cm)', hue='species')
plt.title('Sepal Length vs Sepal Width')
plt.grid(True)
plt.figure(figsize=(8, 6)) # Petal Length vs Petal Width
sns.scatterplot(data=dataframe, x='petal length (cm)', y='petal width (cm)', hue='species')
plt.title('Petal Length vs Petal Width')
plt.grid(True)
plt.figure(figsize=(8, 6)) # Sepal Length vs Petal Length
sns.scatterplot(data=dataframe, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.grid(True)
plt.figure(figsize=(8, 6)) # Sepal Width vs Petal Width
sns.scatterplot(data=dataframe, x='sepal width (cm)', y='petal width (cm)', hue='species')
plt.title('Sepal Width vs Petal Width')
plt.grid(True)
plt.legend(title= "Feature classification") # to check this one
plt.show()

# TASK num. 5: Pattern Recognition/ Data Preprocessing
X = dataframe[iris.feature_names]  # The 4 measurement columns
y = dataframe['species']           # The species column (target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, # it makes the split always the same when you rerun
    stratify=y       # it makes sure the species are evenly balanced in both sets
) # Data split into 80% training and 20% testing
scaler = StandardScaler() # it creates a tool that will scale (normalize) the feature values
# Fit on training data and transform both train and test
X_train_scaled = scaler.fit_transform(X_train) # it fits the scaler to the training data and scales it
X_test_scaled = scaler.transform(X_test) # it scales the test data using the same parameters as the training data


print("--- Ottimizzazione per Logistic Regression ---")
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100], 
    'solver': ['liblinear', 'lbfgs'] 
}
grid_search_lr = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=200),
    param_grid_lr,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
    scoring='accuracy',
    n_jobs=-1, 
    verbose=1 
)
grid_search_lr.fit(X_train_scaled, y_train)

print(f"Migliori parametri per Logistic Regression: {grid_search_lr.best_params_}")
print(f"Migliore accuracy media in CV per Logistic Regression: {grid_search_lr.best_score_:.4f}")
print(f"Accuracy sul test set con i migliori parametri: {accuracy_score(y_test, grid_search_lr.best_estimator_.predict(X_test_scaled)):.4f}")

print("\n--- Ottimizzazione per Random Forest (Randomized Search) ---")
param_distributions_rf = {
    'n_estimators': randint(50, 200), 
    'max_depth': randint(3, 15),     
    'min_samples_leaf': randint(1, 10), 
    'criterion': ['gini', 'entropy'] 
}
random_search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions_rf,
    n_iter=50, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search_rf.fit(X_train_scaled, y_train)

print(f"Migliori parametri per Random Forest: {random_search_rf.best_params_}")
print(f"Migliore accuracy media in CV per Random Forest: {random_search_rf.best_score_:.4f}")
print(f"Accuracy sul test set con i migliori parametri: {accuracy_score(y_test, random_search_rf.best_estimator_.predict(X_test_scaled)):.4f}")

print("\n--- Ottimizzazione per Support Vector Machine ---")
param_grid_svc = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1], 
    'kernel': ['rbf', 'linear'] 
}
grid_search_svc = GridSearchCV(
    SVC(random_state=42),
    param_grid_svc,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search_svc.fit(X_train_scaled, y_train)

print(f"Migliori parametri per SVC: {grid_search_svc.best_params_}")
print(f"Migliore accuracy media in CV per SVC: {grid_search_svc.best_score_:.4f}")
print(f"Accuracy sul test set con i migliori parametri: {accuracy_score(y_test, grid_search_svc.best_estimator_.predict(X_test_scaled)):.4f}")


models_optimized = {
    'Logistic Regression (Optimized)': grid_search_lr.best_estimator_,
    'Decision Tree (Optimized)': DecisionTreeClassifier(random_state=42), # Placeholder, you'd optimize this too
    'Random Forest (Optimized)': random_search_rf.best_estimator_,
    'Support Vector Machine (Optimized)': grid_search_svc.best_estimator_,
    'K-Nearest Neighbors (Optimized)': KNeighborsClassifier(n_neighbors=3) # Placeholder
}

# TASK num. 7: Model Selection, comparison and training process
class ModelComparison:
    def __init__(self, models_optimized):
        self.models = models_optimized
        self.results = {}
        self.trained_models_optimized = {}
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        "Comprehensive model evaluation"
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            self.trained_models_optimized[name] = model
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
    
    def plot_comparison(self):
        "Visualize model comparison"
        df = pd.DataFrame(self.results).T
        plt.figure(figsize=(12, 6))
        df.plot(kind='bar', ax=plt.gca())
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def get_best_model(self):
        """Return the best performing model"""
        df = pd.DataFrame(self.results).T
        best_model_name = df['accuracy'].idxmax()
        return best_model_name, self.trained_models_optimized[best_model_name]

# Usage
comparator = ModelComparison(models_optimized)
comparator.evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
comparator.plot_comparison() 

best_name, best_model = comparator.get_best_model()
print(f"Best model: {best_name}")

# TASK num. 8: Cross-validation and hyperparameter tuning
class CrossValidationAnalyzer:
    def __init__(self, models_optimized, cv_strategy='stratified_kfold', n_splits=5, 
                 random_state=42, shuffle=True):
        self.models_optimized = models_optimized
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        
        # Results storage
        self.cv_results = {}
        self.detailed_results = {}
        self.fold_results = {}
        
        # Setup CV strategy
        self.cv_splitter = self._setup_cv_strategy()

    def _setup_cv_strategy(self):
        """Setup the cross-validation splitting strategy"""
        if self.cv_strategy == 'kfold':
            return KFold(
                n_splits=self.n_splits, 
                shuffle=self.shuffle, 
                random_state=self.random_state
            )
        elif self.cv_strategy == 'stratified_kfold':
            return StratifiedKFold(
                n_splits=self.n_splits, 
                shuffle=self.shuffle, 
                random_state=self.random_state
            )
        elif self.cv_strategy == 'leave_one_out':
            return LeaveOneOut()
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
    
    def perform_cv(self, X, y, scoring_metrics=None):
        if scoring_metrics is None:
            scoring_metrics = {
                'accuracy': 'accuracy',
                'precision': 'precision_weighted',
                'recall': 'recall_weighted',
                'f1': 'f1_weighted'
            }
        print(f"Performing {self.cv_strategy} cross-validation...")
        print(f"CV Strategy: {self.n_splits}-fold" if self.n_splits else "Leave-One-Out")
        print("-" * 50)
        
        for model_name, model in self.models_optimized.items():
            print(f"Evaluating {model_name}...")
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y, 
                cv=self.cv_splitter,
                scoring=scoring_metrics,
                return_train_score=True,
                return_estimator=True
            )
            
            # Store detailed results
            self.detailed_results[model_name] = cv_results
            
            # Calculate summary statistics
            summary = {}
            for metric in scoring_metrics.keys():
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                summary[metric] = {
                    'test_mean': test_scores.mean(),
                    'test_std': test_scores.std(),
                    'test_min': test_scores.min(),
                    'test_max': test_scores.max(),
                    'train_mean': train_scores.mean(),
                    'train_std': train_scores.std(),
                    'overfitting_gap': train_scores.mean() - test_scores.mean()
                }
            
            self.cv_results[model_name] = summary
            
            # Store fold-by-fold results for detailed analysis
            self.fold_results[model_name] = {
                metric: cv_results[f'test_{metric}'] 
                for metric in scoring_metrics.keys()
            }
        
        print("Cross-validation completed!")
        return self.cv_results
    
    def get_cv_summary(self, metric='accuracy'):
        """Get summary DataFrame of CV results"""
        summary_data = []
        
        for model_name, results in self.cv_results.items():
            if metric in results:
                summary_data.append({
                    'Model': model_name,
                    'Mean': results[metric]['test_mean'],
                    'Std': results[metric]['test_std'],
                    'Min': results[metric]['test_min'],
                    'Max': results[metric]['test_max'],
                    'Overfitting_Gap': results[metric]['overfitting_gap']
                })
        
        return pd.DataFrame(summary_data).sort_values('Mean', ascending=False)
    
    def get_best_model(self, metric='accuracy'):
        """Return best model name based on CV results"""
        summary_df = self.get_cv_summary(metric)
        return summary_df.iloc[0]['Model']
    
    def plot_cv_results(self, metric='accuracy', figsize=(12, 6)):
        """Visualize cross-validation results"""
        # Prepare data for plotting
        plot_data = []
        for model_name, fold_scores in self.fold_results.items():
            if metric in fold_scores:
                for fold, score in enumerate(fold_scores[metric]):
                    plot_data.append({
                        'Model': model_name,
                        'Fold': fold + 1,
                        'Score': score
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Boxplot
        sns.boxplot(data=plot_df, x='Model', y='Score', ax=axes[0])
        axes[0].set_title(f'{metric.capitalize()} Distribution Across Folds')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Line plot showing fold-by-fold performance
        for model_name in self.fold_results.keys():
            if metric in self.fold_results[model_name]:
                scores = self.fold_results[model_name][metric]
                axes[1].plot(range(1, len(scores) + 1), scores, 
                           marker='o', label=model_name)
        
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel(f'{metric.capitalize()} Score')
        axes[1].set_title(f'{metric.capitalize()} by Fold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_overfitting_analysis(self, figsize=(10, 6)):
        """Plot training vs validation scores to detect overfitting"""
        models = list(self.cv_results.keys())
        metrics = list(self.cv_results[models[0]].keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            train_means = [self.cv_results[model][metric]['train_mean'] for model in models]
            test_means = [self.cv_results[model][metric]['test_mean'] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[idx].bar(x - width/2, train_means, width, label='Train', alpha=0.7)
            axes[idx].bar(x + width/2, test_means, width, label='Validation', alpha=0.7)
            
            axes[idx].set_xlabel('Models')
            axes[idx].set_ylabel(f'{metric.capitalize()} Score')
            axes[idx].set_title(f'{metric.capitalize()}: Train vs Validation')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(models, rotation=45)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_statistical_significance(self, model1, model2, metric='accuracy', alpha=0.05):
        """Test statistical significance between two models"""
        from scipy import stats
        
        scores1 = self.fold_results[model1][metric]
        scores2 = self.fold_results[model2][metric]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        result = {
            'model1': model1,
            'model2': model2,
            'metric': metric,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'better_model': model1 if np.mean(scores1) > np.mean(scores2) else model2
        }
        
        return result
    
    def export_results(self, filename='cv_results.csv'):
        """Export CV results to CSV"""
        all_results = []
        
        for model_name, results in self.cv_results.items():
            for metric, stats in results.items():
                all_results.append({
                    'Model': model_name,
                    'Metric': metric,
                    **stats
                })
        
        dataframe = pd.DataFrame(all_results)
        dataframe.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

    def plot_roc_curves(self, figsize=(10, 8)):
        if not self.roc_auc_data:
            print("Nessun dato ROC/AUC disponibile. Eseguire evaluate_models prima.")
            return

        plt.figure(figsize=figsize)
        colors = plt.cm.get_cmap('tab10', len(self.roc_auc_data)) 

        for i, (model_name, data) in enumerate(self.roc_auc_data.items()):
            # Plot Micro-average ROC
            plt.plot(data['fpr']['micro'], data['tpr']['micro'],
                     label=f'{model_name} (Micro-avg AUC = {data["roc_auc"]["micro"]:.2f})',
                     color=colors(i), linestyle='-', linewidth=2) 

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Classificatore Casuale (AUC = 0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC Multiclasse (Micro-average)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        for model_name, data in self.roc_auc_data.items():
            print(f"\n--- Dettagli AUC per {model_name} ---")
            for j, class_name in enumerate(data['class_names']):
                print(f"  Classe {class_name}: AUC = {data['roc_auc'][j]:.4f}")
            print(f" Micro-average AUC: {data['roc_auc']['micro']:.4f}")
            print(f" Macro-average AUC: {data['roc_auc']['macro']:.4f}")
            
            plt.figure(figsize=(8, 6))
            for j, class_name in enumerate(data['class_names']):
                 plt.plot(data['fpr'][j], data['tpr'][j],
                          label=f'{class_name} (AUC = {data["roc_auc"][j]:.2f})',
                          linestyle='-', linewidth=1.5)
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Curva ROC Per-Class per {model_name}')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()

analyzer = CrossValidationAnalyzer(models_optimized) 
cv_results = analyzer.perform_cv(X_train_scaled, y_train)
analyzer.export_results('cv_results.csv')
print(f"File saved as: cv_results.csv")
if os.path.exists('cv_results.csv'):
    print("✓ File created with success!")