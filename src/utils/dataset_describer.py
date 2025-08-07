import pandas as pd
import numpy as np
from scipy.stats import zscore

def analyze_outliers(df, rare_threshold=0.05, z_thresh=3):
    results = []

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        series = df[col]
        
        z_scores = zscore(series)
        z_outliers = np.where(np.abs(z_scores) > z_thresh)[0]

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]

        total_outliers = np.unique(np.concatenate((z_outliers, iqr_outliers.index)))
        num_outliers = len(total_outliers)
        pct_outliers = num_outliers / len(series)

        if series.min() < 0 and col in ['use_count', 'average_basket_size', 'membership_fee']:  
            classification = "Obvious error"  
            action = "Fix or remove"  
        elif pct_outliers == 0:  
            classification = "No outliers detected"  
            action = "No action needed"  
        elif pct_outliers < rare_threshold:  
            classification = "Rare but possible"  
            action = "Keep or segment"  
        elif pct_outliers >= rare_threshold:  
            classification = "May harm model"  
            action = "Transform or use robust model"  
        else:  
            classification = "Unclassified"  
            action = "Manual review required"  

        results.append({  
            "column": col,  
            "outliers_detected": num_outliers,  
            "percentage": round(pct_outliers * 100, 2),  
            "classification": classification,  
            "suggested_action": action  
        })  

    return pd.DataFrame(results)


def describe_dataset(df: pd.DataFrame, name: str='dataset'):
    """
    Prints an exploratory data analysis (EDA) summary of a given Pandas DataFrame.

    Parameters
    ----------
    df : Pandas DataFrame
        The DataFrame to be analyzed.
    name : str, optional
        The name to be printed as header for the EDA summary. Defaults to 'dataset'.
    """
    print(f"\nEDA for {name}:")
    print("-" * 30)

    print(f"Dimensions: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nData types:")
    print(df.dtypes.value_counts())

    print("\nColumn names:")
    print(df.columns)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())

    print("\nMissing values:")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if nulls.empty:
        print("No missing values found.")
    else:
        print(nulls.to_frame(name='Nulls').assign(Pct=lambda x: (x['Nulls'] / len(df) * 100).round(2)))

    print("\nNumeric summary:")
    print(df.describe(include=[np.number]).T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    print("\nCategorical columns with cardinality:")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        n_unique = df[col].nunique()
        top = df[col].value_counts().index[0]
        freq = df[col].value_counts().iloc[0]
        print(f" - {col}: {n_unique} categories (more common: '{top}' -> {freq})")

    print("\nOutlier verifications:")
    analyze_outliers(df)

    print("\nOther verifications:")
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if constant_cols:
        print(f" - Constant columns: {constant_cols}")
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f" - Duplicate columns: {duplicate_cols}")
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        print(f" - Duplicate rows: {duplicate_rows}")

    print("-" * 30)
    