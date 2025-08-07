import pandas as pd
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_data(df: pd.DataFrame, save_dir: Union[str, Path], base_filename: str = "data", formats: Optional[Dict[str, Dict]] = None, sqlite_table: str = "data_table", timestamp: bool = False) -> Dict[str, Path]:
    """
    Saves DataFrame to multiple formats with automatic extension handling for compression.

    Args:
        df: DataFrame to save
        save_dir: Output directory path
        base_filename: Base filename without extension
        formats: Dictionary with format-specific options. Defaults to:
            {
                "csv": {"index": False, "compression": "gzip"},
                "parquet": {"index": False, "compression": "snappy"},
                "sqlite": {"if_exists": "replace"}
            }
        sqlite_table: Table name for SQLite
        timestamp: Whether to append timestamp to filename

    Returns:
        Dictionary of {"format": path} for saved files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if timestamp:
        from datetime import datetime
        base_filename = f"{base_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    default_formats = {
        "csv": {"index": False, "compression": "gzip"},
        "parquet": {"index": False, "compression": "snappy"},
        "sqlite": {"if_exists": "replace"}
    }
    formats = formats or default_formats
    
    saved_files = {}
    
    try:
        if "csv" in formats:
            compression = formats["csv"].get("compression", None)
            ext = ".csv.gz" if compression in ["gzip", "gz"] else ".csv"
            csv_path = save_dir / f"{base_filename}{ext}"
            df.to_csv(csv_path, **formats["csv"])
            saved_files["csv"] = csv_path
            logger.info(f"Saved to CSV ({'compressed' if compression else 'uncompressed'}): {csv_path}")
        
        if "parquet" in formats:
            parquet_path = save_dir / f"{base_filename}.parquet"
            df.to_parquet(parquet_path, **formats["parquet"])
            saved_files["parquet"] = parquet_path
            logger.info(f"Saved to Parquet: {parquet_path}")
        
        if "sqlite" in formats:
            sqlite_path = save_dir / f"{base_filename}.db"
            conn = sqlite3.connect(sqlite_path)
            df.to_sql(sqlite_table, conn, **formats["sqlite"])
            conn.close()
            saved_files["sqlite"] = sqlite_path
            logger.info(f"Saved to SQLite (table={sqlite_table}): {sqlite_path}")
            
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}", exc_info=True)
        raise
    
    return saved_files