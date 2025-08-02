import pandas as pd
import numpy as np

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