import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def load_data(filepath, target_column):
    """
    Load data from a csv file and split it into training and testing sets.

    Args:
        filepath (str): The path to the csv file.
        target_column (str): The name of the target column in the csv file.

    Returns:
        tuple: A tuple containing the training features (X_train) and target variable (y_train),
               and the testing features (X_test) and target variable (y_test).
    """
    df = pd.read_csv(filepath)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=115)

def train_model(X_train, y_train):
    """
    Train a machine learning model on the given training data.

    Args:
        X_train (array-like): The training features.
        y_train (array-like): The target variable for the training data.

    Returns:
        estimator: The best estimator found by the randomized search.

    The function trains a GradientBoostingClassifier with the given training data
    and performs a randomized search to find the best parameters. The best estimator
    found is returned.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(random_state=115))
    ])

    param_dist = {
        'model__n_estimators': np.arange(50, 501, 50),
        'model__learning_rate': np.linspace(0.01, 0.3, 10),
        'model__max_depth': np.arange(2, 8),
        'model__min_samples_split': np.arange(2, 21),
        'model__min_samples_leaf': np.arange(1, 11),
        'model__subsample': np.linspace(0.6, 1.0, 5),
        'model__max_features': ['sqrt', 'log2', None],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,
        scoring='recall',
        cv=5,
        verbose=2,
        random_state=115,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    print("\nBest parameters found:", search.best_params_)
    return search.best_estimator_

def evaluate_thresholds(model, X_test, y_test, min_precision=0.7):
    """
    Evaluate the performance of a model at different thresholds for binary classification.

    Args:
        model (object): Trained model object.
        X_test (array-like): Test data.
        y_test (array-like): True labels for the test data.
        min_precision (float, optional): Minimum required precision for considering a threshold. Defaults to 0.7.

    Returns:
        metrics_df (pandas.DataFrame): DataFrame containing the performance metrics for different thresholds.
        optimal_threshold (float): Optimal threshold based on maximum recall with precision >= min_precision.
        y_proba (array-like): Predicted probabilities for the test data.

    This function evaluates the performance of a model at different thresholds for binary classification.
    It calculates precision, recall, and F1 score for each threshold and returns the threshold with the highest recall
    while satisfying the minimum precision requirement. It also returns a DataFrame containing the performance metrics
    for different thresholds and the predicted probabilities for the test data.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.0, 1.01, 0.05)
    metrics = []

    for t in thresholds:
        y_pred_thresh = (y_proba >= t).astype(int)
        metrics.append((
            t,
            precision_score(y_test, y_pred_thresh),
            recall_score(y_test, y_pred_thresh),
            f1_score(y_test, y_pred_thresh)
        ))

    metrics_df = pd.DataFrame(metrics, columns=["Threshold", "Precision", "Recall", "F1"])
    valid_thresholds = metrics_df[metrics_df["Precision"] >= min_precision]
    best_threshold_row = valid_thresholds.loc[valid_thresholds['Recall'].idxmax()]

    print(f"\nOptimized threshold (Maximum Recall with Precision >= {min_precision}): {best_threshold_row['Threshold']:.2f}")
    print(f"(Precision={best_threshold_row['Precision']:.3f}, "
          f"Recall={best_threshold_row['Recall']:.3f}, "
          f"F1={best_threshold_row['F1']:.3f})")

    return metrics_df, best_threshold_row['Threshold'], y_proba

def final_evaluation(y_test, y_proba, threshold):
    """
    Compute and print the final evaluation metrics for a binary classification problem.

    Args:
        y_test (array-like): True labels for the test data.
        y_proba (array-like): Predicted probabilities for the test data.
        threshold (float): The threshold to classify a prediction as positive.

    Returns:
        array-like: Predicted labels based on the given threshold.

    This function computes and prints the accuracy, precision, recall, F1 score, and ROC AUC score
    for the given test data and predicted probabilities with the given threshold. It also returns
    the predicted labels based on the threshold.
    """
    y_pred_final = (y_proba >= threshold).astype(int)
    print("\n=== Metrics with optimized threshold ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred_final):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred_final):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred_final):.3f}")
    print(f"F1       : {f1_score(y_test, y_pred_final):.3f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, y_proba):.3f}")
    return y_pred_final

def plot_confusion_matrix(y_test, y_pred, threshold):
    """
    Plots the confusion matrix for a binary classification problem.

    Args:
        y_test (array-like): True labels for the test data.
        y_pred (array-like): Predicted labels for the test data.
        threshold (float): The threshold used to classify a prediction as positive.

    Returns:
        None

    This function plots the confusion matrix for the given test data and predicted labels
    with the specified threshold. The confusion matrix shows the number of true positives,
    false positives, true negatives, and false negatives for different classes.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.title(f"Confusion Matrix - Threshold={threshold:.2f}")
    plt.show()

def plot_metrics_vs_threshold(metrics_df):
    """
    Plot precision, recall and F1-score vs threshold.

    Args:
        metrics_df (pandas.DataFrame): DataFrame containing the metrics and their corresponding thresholds.

    Returns:
        None

    This function plots the precision, recall and F1-score against the threshold. It creates a line plot
    with markers indicating the scores at each threshold. The plot shows the relationship between the
    threshold and the metrics for the class with index 1.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["Threshold"], metrics_df["Precision"], label="Precision", marker="o")
    plt.plot(metrics_df["Threshold"], metrics_df["Recall"], label="Recall", marker="o")
    plt.plot(metrics_df["Threshold"], metrics_df["F1"], label="F1-score", marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall and F1 vs Threshold (Class 1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def save_model(model, filepath):
    """
    Save the trained model to a file.

    Args:
        model (object): Trained model object.
        filepath (str): Path where the model will be saved.

    Returns:
        None

    This function saves the trained model to a file using joblib's dump function.
    It prints the path where the model was saved.
    """
    joblib.dump(model, filepath)
    print(f"Model saved at: {filepath}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(
        filepath="data/processed/processed_cybersecurity_intrusion_data.csv",
        target_column='remainder__attack_detected'
    )

    best_model = train_model(X_train, y_train)

    metrics_df, best_threshold, y_proba = evaluate_thresholds(best_model, X_test, y_test)

    y_pred_final = final_evaluation(y_test, y_proba, best_threshold)

    plot_confusion_matrix(y_test, y_pred_final, best_threshold)
    plot_metrics_vs_threshold(metrics_df)

    save_model(best_model, "models/final_model.joblib")