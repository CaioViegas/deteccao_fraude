import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.paths import get_project_paths

class DataTransformer:
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
                self.issues.append(f"Column: {col} has constant values: {self.df[col].iloc[0]}")

    def detect_high_cardinality(self, threshold: int=50):
        for col in self.df.select_dtypes(include='object').columns:
            unique_vals = self.df[col].nunique()
            if unique_vals > threshold:
                self.issues.append(f"Column: {col} has high cardinality: ({unique_vals} unique values)")

    def detect_date_issues(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    parsed = pd.to_datetime(self.df[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                    if parsed.isnull().sum() < len(self.df) * 0.5:
                        self.df[col] = parsed
                    else:
                        continue
                
                except Exception:
                    continue

            if np.issubdtype(self.df[col].dtype, np.datetime64):
                nulls = self.df[col].isnull().sum()
                if nulls > 0:
                    self.issues.append(f"Datetime columns '{col}' has {nulls} missing values.")

    def transform(self):
        self.impute_missing_values()
        self.detect_constant_columns()
        self.detect_high_cardinality()
        self.detect_date_issues()           
        return self.df
    
    def report_issues(self):
        if not self.issues:
            print("No issues found in the DataFrame.")

        else:
            print("Issues found in the DataFrame:")
            for issue in self.issues:
                print(issue)

class OutlierHandler:
    def __init__(self, df: pd.DataFrame, outlier_log: Path=None):
        self.df = df.copy()
        self.outliers_detected = pd.DataFrame()
        self.log_dir = outlier_log
        self.issues = []

    def detect_iqr_outliers(self, multiplier=1.5):
        numeric_cols = self.df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr

            mask = (self.df[col] < lower) | (self.df[col] > upper)
            count = mask.sum()
            if count > 0:
                self.df[f"outlier_{col}"] = mask.astype(int)
                self.outliers_detected = pd.concat([self.outliers_detected, self.df[mask]])
                self.issues.append(f"{count} outliers detected in column '{col}' using IQR")

    def segment_extreme_values(self, column: str, upper_threshold: float):
        mask = self.df[column] > upper_threshold
        count = mask.sum()
        if count > 0:
            self.df[f"is_extreme_{column}"] = mask.astype(int)
            self.issues.append(f"{count} extreme values detected in column '{column}'")

    def export_log(self, filename='outliers.csv'):
        if self.outliers_detected.empty or self.log_dir is None:
            return
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / filename
        self.outliers_detected.to_csv(path, index=False)
        self.issues.append(f"Outliers exported to {path}")

    def report(self):
        if not self.issues:
            print("No issues found in the DataFrame.")
        else:
            print("\nOutliers report")
            for issue in self.issues:
                print(issue)

    def get_dataframe(self):
        return self.df

def run_transform():
    paths = get_project_paths()
    raw_dir = paths['RAW']
    transformed_dir = paths['TRANSFORMED']
    logs_dir = paths['LOGS'] / "outliers"
    transformed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        print("No csv files found in the raw directory.")
        return

    file_path = csv_files[0]
    print(f"Transforming file: {file_path.name}")
    df = pd.read_csv(file_path)

    outlier_handler = OutlierHandler(df, outlier_log=logs_dir)
    outlier_handler.detect_iqr_outliers()
    outlier_handler.segment_extreme_values(column='Transaction_Amount', upper_threshold=1000)  # exemplo específico
    outlier_handler.export_log()
    outlier_handler.report()

    df_cleaned = outlier_handler.get_dataframe()
    transformer = DataTransformer(df_cleaned)
    df_transformed = transformer.transform()
    transformer.report_issues()

    output_path = transformed_dir / f"transformed_{file_path.name}"
    df_transformed.to_csv(output_path, index=False)
    print(f"\nTransformed file saved: {output_path}")

if __name__ == "__main__":    
    run_transform()