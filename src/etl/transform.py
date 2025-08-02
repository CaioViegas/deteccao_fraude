import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class DataTransfomer:
    def __init__(self, df: pd.DataFrame, knn_neighbors: int=5):
        self.df = df.copy()
        self.knn_neighbors = knn_neighbors
        self.issues = []

    def impute_missing_values(self):
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if numeric_cols.empty:
            self.issues.append("No numeric columns found in the DataFrame.")
            return
        
        missing_before = self.df[numeric_cols].isnull().sum()
        total_missing = missing_before.sum()

        if total_missing == 0:
            return
        
        imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        imputed_data = imputer.fit_transform(self.df[numeric_cols])
        self.df[numeric_cols] = pd.DataFrame(imputed_data, columns=numeric_cols)

        for col in numeric_cols:
            if missing_before[col] > 0:
                self.issues.append(f"Imputed {missing_before[col]} missing values in column {col}.")

    def detect_constant_columns(self):
        for col in self.df.columns:
            if self.df[col].nunique(dropna=False) == 1:
                self.issues.append