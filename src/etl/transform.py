import sys
import pandas as pd
import numpy as np
from load import save_data
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
        """
        Imputes missing values in numeric columns using K-Nearest Neighbors imputation.
        """
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
        """
        Detects columns in the DataFrame that have constant values 
        and logs them as issues.

        A column is considered constant if it has only one unique value, 
        including NaNs. The detected constant columns are appended to the 
        issues list with their respective constant value.
        """
        for col in self.df.columns:
            if self.df[col].nunique(dropna=False) == 1:
                self.issues.append(f"Column: {col} has constant values: {self.df[col].iloc[0]}")

    def detect_high_cardinality(self, threshold: int=50):
        """
        Detects columns in the DataFrame that have high cardinality (i.e., too many unique values) 
        and logs them as issues.

        A column is considered to have high cardinality if it has more unique values than the given threshold.
        The detected columns are appended to the issues list with their respective number of unique values.
        """
        for col in self.df.select_dtypes(include='object').columns:
            unique_vals = self.df[col].nunique()
            if unique_vals > threshold:
                self.issues.append(f"Column: {col} has high cardinality: ({unique_vals} unique values)")

    def detect_date_issues(self):
        """
        Detects date columns with missing values and logs them as issues.

        Checks if date columns have missing values and logs them as issues with their respective number of missing values.
        If the column is of object type, it tries to parse it as datetime using the format '%Y-%m-%d %H:%M:%S' and if it succeeds, it replaces the column with the parsed datetime values.
        If the column is of datetime type, it checks if it has missing values and logs it as an issue if it does.
        """
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
        """
        Transforms the DataFrame by performing the following operations:

        - Imputes missing values in numeric columns using K-Nearest Neighbors imputation.
        - Detects columns with constant values and logs them as issues.
        - Detects columns with high cardinality (i.e., too many unique values) and logs them as issues.
        - Detects date columns with missing values and logs them as issues.

        Returns the transformed DataFrame.
        """
        self.impute_missing_values()
        self.detect_constant_columns()
        self.detect_high_cardinality()
        self.detect_date_issues()           
        return self.df
    
    def report_issues(self):
        """
        Reports any issues found in the DataFrame after running the transform method.

        Prints a message indicating if any issues were found in the DataFrame. If issues were found, it prints each issue as a separate line.
        """
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
        """
        Detects outliers in numeric columns of the DataFrame using the Interquartile Range (IQR) method.

        The method calculates the first (Q1) and third (Q3) quartiles for each numeric column, and determines 
        the IQR as the difference between Q3 and Q1. Outliers are defined as values that fall below Q1 - (multiplier * IQR) 
        or above Q3 + (multiplier * IQR). For columns with detected outliers, a new column is created indicating 
        outlier presence, and details are logged.

        Args:
            multiplier (float, optional): The multiplier for the IQR to define the bounds for outliers. 
                                        Defaults to 1.5, which is a common choice for identifying outliers.

        Logs:
            The number of outliers detected for each column is appended to the issues list.

        Updates:
            - Adds a binary column for each numeric column indicating the presence of outliers.
            - Concatenates the rows with outliers to the outliers_detected DataFrame.
        """
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
        """
        Segments extreme values in a specified column by marking them in a new column.

        This method identifies values in the specified column that are greater than the given upper threshold. 
        For each identified extreme value, a new binary column is created to indicate the presence of extreme values, 
        and a log entry is recorded with the count of extreme values detected.

        Args:
            column (str): The name of the column to check for extreme values.
            upper_threshold (float): The threshold above which values are considered extreme.

        Updates:
            - Adds a binary column indicating extreme values for the specified column.
            - Logs the count of extreme values detected in the issues list.
        """
        mask = self.df[column] > upper_threshold
        count = mask.sum()
        if count > 0:
            self.df[f"is_extreme_{column}"] = mask.astype(int)
            self.issues.append(f"{count} extreme values detected in column '{column}'")

    def export_log(self, filename='outliers.csv'):
        """
        Exports the outliers detected in the DataFrame to a CSV file.

        The method saves the outliers_detected DataFrame to a CSV file in the specified log directory.

        Args:
            filename (str, optional): The name of the CSV file to be saved. Defaults to 'outliers.csv'.
        """
        if self.outliers_detected.empty or self.log_dir is None:
            return
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / filename
        self.outliers_detected.to_csv(path, index=False)
        self.issues.append(f"Outliers exported to {path}")

    def report(self):
        """
        Reports any issues found in the DataFrame after running the transform method.

        Prints a message indicating if any issues were found in the DataFrame. If issues were found, it prints each issue as a separate line.
        """
        if not self.issues:
            print("No issues found in the DataFrame.")
        else:
            print("\nOutliers report")
            for issue in self.issues:
                print(issue)

    def get_dataframe(self):
        return self.df

def run_transform():
    """
    Runs the outlier detection and data transformation pipeline.

    This function runs the following steps:

    1. Loads a CSV file from the raw directory.
    2. Detects outliers in the DataFrame using the IQR method.
    3. Segments extreme values in a specified column.
    4. Exports the outliers detected to a CSV file in the logs directory.
    5. Reports any issues found in the DataFrame after running the transform method.
    6. Transforms the DataFrame using the DataTransformer class.
    7. Reports any issues found in the transformed DataFrame.
    8. Saves the transformed DataFrame to a CSV file in the transformed directory.
    """
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

    base_filename = f"transformed_{file_path.stem}"
    save_data(df_transformed, save_dir=transformed_dir, base_filename=base_filename)
    print(f"\nTransformed file saved")

if __name__ == "__main__":    
    run_transform()