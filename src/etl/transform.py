import sys
import pandas as pd
import numpy as np
import unicodedata
import logging
from load import save_data
from pathlib import Path
from sklearn.impute import KNNImputer
from typing import Dict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.paths import get_project_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataTransformer:
    def __init__(self, data: pd.DataFrame, knn_neighbors: int = 5):
        self.data = data
        self.knn_neighbors = knn_neighbors
        self.imputer = KNNImputer(n_neighbors=knn_neighbors)
        self.metadata: Dict = {}

    def remove_duplicates(self) -> 'DataTransformer':
        """
        Removes duplicate rows from the DataFrame.

        Tracks the number of removed duplicate rows in the metadata attribute
        and logs the count using the logger.

        Returns:
            DataTransformer: The instance of the DataTransformer class.
        """
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        removed_rows = initial_rows - len(self.data)

        self.metadata['duplicates_removed'] = removed_rows
        logger.info(f"Removed {removed_rows} duplicate rows.")
        return self
    
    def handle_missing(self) -> 'DataTransformer':
        """
        Handles missing values in the DataFrame.

        For numeric columns, uses K-Nearest Neighbors Imputation to fill the gaps.

        For categorical columns, fills missing values with the mode of the column. If the column has no mode, it fills it with 'missing'.
        """
        num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()

        if num_cols:
            imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            before_na = self.data[num_cols].isna().sum().sum()
            self.data[num_cols] = imputer.fit_transform(self.data[num_cols])
            after_na = self.data[num_cols].isna().sum().sum()
            logger.info(f"KNN Imputer filled {before_na - after_na} missing numeric values.")

        for col in cat_cols:

            if self.data[col].isna().any():
                mode = self.data[col].mode()
                if not mode.empty:
                    fill_value = mode[0]
                    self.data[col].fillna(fill_value, inplace=True)
                    logger.info(f"Filled missing categorical '{col}' with mode: {fill_value}")

                else:
                    self.data[col].fillna('missing', inplace=True)
                    logger.info(f"Filled missing categorical '{col}' with 'missing'")
        return self
    
    def clean_text_data(self) -> 'DataTransformer':
        """
        Cleans the text data in the DataFrame.

        Strips whitespace from the text columns, and for the protocol_type, encryption_used, and browser_type columns, converts them to uppercase.

        Uses unicodedata.normalize('NFKD', ..) to remove any accents and special characters from the text columns.

        Returns:
            DataTransformer: The instance of the DataTransformer class.
        """
        text_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            if col in ['session_id', 'protocol_type', 'encryption_used', 'browser_type']:
                self.data[col] = self.data[col].str.strip()
                
                if col != 'session_id':
                    self.data[col] = self.data[col].str.upper()
                
                self.data[col] = self.data[col].apply(
                    lambda x: unicodedata.normalize('NFKD', str(x))
                    .encode('ASCII', 'ignore')
                    .decode('ASCII')
                )
        
        return self
    
    def correct_typo_and_inconsistencies(self) -> 'DataTransformer':
        """
        Corrects typos and inconsistencies in the DataFrame.

        For the columns 'protocol_type', 'encryption_used', and 'browser_type', the function
        standardizes the values by upper-casing and stripping whitespace. It then filters
        out values that are not in the allowed sets.

        The allowed sets are:
        - 'protocol_type': {'TCP', 'UDP', 'ICMP'}
        - 'encryption_used': {'AES', 'DES'}
        - 'browser_type': {'Chrome', 'Firefox', 'Edge', 'Safari', 'Opera'}

        Logs the number of filtered rows for each column.

        Returns:
            DataTransformer: The instance of the DataTransformer class.
        """
        if 'protocol_type' in self.data.columns:
            self.data['protocol_type'] = self.data['protocol_type'].str.upper().str.strip()
            allowed_protocols = {'TCP', 'UDP', 'ICMP'}
            self.data = self.data[self.data['protocol_type'].isin(allowed_protocols)]
            logger.info("Standardized and filtered 'protocol_type'.")
        
        if 'encryption_used' in self.data.columns:
            self.data['encryption_used'] = self.data['encryption_used'].str.upper().str.strip()
            allowed_encryptions = {'AES', 'DES'}
            self.data = self.data[self.data['encryption_used'].isin(allowed_encryptions)]
            logger.info("Standardized and filtered 'encryption_used'.")
        
        if 'browser_type' in self.data.columns:
            self.data['browser_type'] = self.data['browser_type'].str.title().str.strip()
            allowed_browsers = {'Chrome', 'Firefox', 'Edge', 'Safari', 'Opera'}
            self.data = self.data[self.data['browser_type'].isin(allowed_browsers)]
            logger.info("Standardized and filtered 'browser_type'.")
        
        return self
    
    def filter_invalid_data(self) -> 'DataTransformer':
        """
        Filters invalid data in the DataFrame.

        Currently, this function only filters out rows with invalid 'session_duration' values (i.e. <= 0).

        Logs the number of rows removed for each column.

        Returns:
            DataTransformer: The instance of the DataTransformer class.
        """
        if 'session_duration' in self.data.columns:
            before = self.data.shape[0]
            self.data = self.data[self.data['session_duration'] > 0]
            after = self.data.shape[0]
            logger.info(f"Filtered invalid 'session_duration' rows: removed {before - after} rows.")
        return self
        
    def convert_data_types(self) -> 'DataTransformer':
        """
        Converts data types of numeric columns to numeric types.

        Currently, this function only converts 'session_duration', 'login_attempts', and 'attack_detected' columns.

        Logs the number of columns converted.

        Returns:
            DataTransformer: The instance of the DataTransformer class.
        """
        if 'session_duration' in self.data.columns:
            self.data['session_duration'] = pd.to_numeric(self.data['session_duration'], errors='coerce')

        if 'login_attempts' in self.data.columns:
            self.data['login_attempts'] = pd.to_numeric(self.data['login_attempts'], downcast='integer', errors='coerce')

        if 'attack_detected' in self.data.columns:
            self.data['attack_detected'] = self.data['attack_detected'].astype('int')

        logger.info("Converted data types for numeric columns.")
        return self
    
    def create_features(self) -> 'DataTransformer':
        """
        Creates new features for the DataFrame.

        - Adds a 'long_session' column indicating if a session's duration is greater than the median session duration.
        - Combines 'protocol_type' and 'encryption_used' into a new 'protocol_encrypt' feature.
        - Adds a 'large_packet' column indicating if a network packet size is greater than the mean packet size.
        - Computes a 'risk_score' as a weighted sum of 'ip_reputation_score' and 'failed_logins'.

        Returns:
            DataTransformer: The instance of the DataTransformer class.
        """
        median_duration = self.data['session_duration'].median()
        self.data['long_session'] = (
            self.data['session_duration'] > median_duration
        ).astype(int)
        
        self.data['protocol_encrypt'] = (
            self.data['protocol_type'].astype(str) + "_" + 
            self.data['encryption_used'].astype(str)
        )
        
        mean_packet = self.data['network_packet_size'].mean()
        self.data['large_packet'] = (
            self.data['network_packet_size'] > mean_packet
        ).astype(int)
        
        self.data['risk_score'] = (
            self.data['ip_reputation_score'] * 0.7 + 
            self.data['failed_logins'] * 0.3
        )
        
        return self
    
    def process_datetime_columns(self) -> 'DataTransformer':
        """
        Processes datetime columns in the DataFrame by converting them to datetime objects and
        extracting relevant temporal features.

        Identifies columns containing 'date' or 'time' in their names and attempts to convert
        them from strings to datetime objects. For successfully converted columns, it extracts
        the year, month, day, weekday, and quarter into separate features.

        Additionally, if 'start_time' and 'end_time' columns exist, it computes the difference
        between them in minutes and adds it as a new feature 'session_time_diff_minutes'.

        Returns:
            DataTransformer: The instance of the DataTransformer class.
        """
        datetime_cols = [
            col for col in self.data.columns 
            if ('date' in col.lower() or 'time' in col.lower()) 
            and self.data[col].dtype == 'object'
        ]

        for col in datetime_cols:
            try:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
            except Exception:
                continue  

            self.data[f'{col}_year'] = self.data[col].dt.year
            self.data[f'{col}_month'] = self.data[col].dt.month
            self.data[f'{col}_day'] = self.data[col].dt.day
            self.data[f'{col}_weekday'] = self.data[col].dt.weekday
            self.data[f'{col}_quarter'] = self.data[col].dt.quarter

        if 'start_time' in self.data.columns and 'end_time' in self.data.columns:
            self.data['session_time_diff_minutes'] = (
                self.data['end_time'] - self.data['start_time']
            ).dt.total_seconds() / 60

        return self
    
    def encode_categorical_variables(self) -> 'DataTransformer':
        """
        Encodes categorical variables in the DataFrame using LabelEncoder for binary features
        and OneHotEncoder for multi-class features.

        Drops the 'session_id' column.

        Identifies categorical columns and separates them into binary and multi-class features.

        Uses LabelEncoder to encode binary features and OneHotEncoder to encode multi-class
        features.

        Returns:
            DataTransformer: The instance of the DataTransformer class.
        """
        self.data.drop(columns=['session_id'], inplace=True)
        cat_cols = self.data.select_dtypes(include='object').columns.tolist()

        binary_cols = [col for col in cat_cols if self.data[col].nunique() == 2]
        for col in binary_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            cat_cols.remove(col)

        if cat_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ct = ColumnTransformer(
                transformers=[('ohe', ohe, cat_cols)],
                remainder='passthrough'
            )
            transformed = ct.fit_transform(self.data)
            feature_names = ct.get_feature_names_out()
            self.data = pd.DataFrame(transformed, columns=feature_names, index=self.data.index)

        return self

    def handle_outliers(self) -> 'DataTransformer':
        """
        Handles outliers in the DataFrame by capping them using the IQR method and
        adding a general outlier flag column. Additionally, it applies a log
        transformation to the 'session_duration' column.

        Outliers are capped by computing the interquartile range (IQR) for each numeric
        column and setting values outside 1.5 times the IQR to the nearest bound. The
        outlier flag is set to 1 if any column contains an outlier.

        Returns:
            DataTransformer: The instance of the DataTransformer class.
        """
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        outlier_flags = pd.Series(0, index=self.data.index)

        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outlier_flags |= ((self.data[col] < lower) | (self.data[col] > upper))

            self.data[col] = np.clip(self.data[col], lower, upper)

        self.data['outlier_flag'] = outlier_flags.astype(int)

        if 'session_duration' in self.data.columns:
            self.data['session_duration'] = np.log1p(self.data['session_duration'])

        logger.info("Outliers capped and log-transformed where necessary.")
        return self
    
    def transform(self) -> pd.DataFrame:
        """
        Applies all the transformations to the data and saves the transformed DataFrame to file.

        This method applies all the transformations to the data, including removing duplicates, handling missing values, encoding categorical variables, and handling outliers. It then saves the transformed DataFrame to file in the 'transformed' directory.

        After that, it applies additional transformations, such as encoding categorical variables and handling outliers, and saves the resulting DataFrame to file in the 'processed' directory.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        paths = get_project_paths()
        processed_dir = paths['PROCESSED']
        transformed_dir = paths['TRANSFORMED']

        formats = {
            "csv": {"index": False, "compression": None},
            "parquet": {"index": False, "compression": "snappy"},
            "sqlite": {"if_exists": "replace"}
        }

        self.remove_duplicates()\
            .clean_text_data()\
            .correct_typo_and_inconsistencies()\
            .filter_invalid_data()\
            .convert_data_types()\
            .handle_missing()\
            .create_features()\
            .process_datetime_columns()

        save_data(self.data, transformed_dir, "transformed_cybersecurity_intrusion_data", formats=formats)

        self.encode_categorical_variables()\
            .handle_outliers()

        save_data(self.data, processed_dir, "processed_cybersecurity_intrusion_data", formats=formats)

        return self.data
    
if __name__ == "__main__":
    paths = get_project_paths()
    raw_dir = paths['RAW']
    df = pd.read_csv(raw_dir / "cybersecurity_intrusion_data.csv")
    transformer = DataTransformer(df)
    transformer.transform()