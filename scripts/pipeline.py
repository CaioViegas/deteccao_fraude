import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.etl.extract import download_dataset
from src.etl.transform import DataTransformer
from configs.paths import get_project_paths

if __name__ == "__main__":
    download_dataset(
        "dnkumars/cybersecurity-intrusion-detection-dataset",
        file_extension=".csv"
    )

    paths = get_project_paths()
    raw_dir = paths['RAW']
    df = pd.read_csv(raw_dir / "cybersecurity_intrusion_data.csv")

    transformer = DataTransformer(df)
    transformer.transform()