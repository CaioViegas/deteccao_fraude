import pandas as pd
import sqlite3
from pathlib import Path

def save_data(df: pd.DataFrame, save_dir: Path, base_filename: str="data"):
    """
    Saves the given DataFrame to a CSV, Parquet file, and SQLite DB at the given directory.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        save_dir (Path): The directory where the files will be saved. The directory will be created if it does not exist.
        base_filename (str, optional): The base filename to be used for the saved files. Defaults to "data".

    Returns:
        None
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / f"{base_filename}.csv"
    parquet_path = save_dir / f"{base_filename}.parquet"
    sqlite_path = save_dir / f"{base_filename}.db"

    df.to_csv(csv_path, index=False)
    print(f"Saved to CSV: {csv_path}")

    df.to_parquet(parquet_path, index=False)
    print(f"Saved to Parquet: {parquet_path}")

    conn = sqlite3.connect(sqlite_path)
    df.to_sql("data_table", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved to SQLite DB: {sqlite_path}")