import sys
import kagglehub
import shutil
import pandas as pd
from pathlib import Path
from load import save_data

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.paths import get_project_paths
from src.utils.dataset_describer import describe_dataset

def download_dataset(slug: str, file_extension: str = ".csv") -> list[Path]:
    """
    Downloads a Kaggle dataset and saves it in the 'raw' directory.

    It will download the dataset with the given slug, copy all files with the given extension to the 'raw' directory and
    save a parquet and SQLite version of each file.

    Args:
        slug (str): The slug of the dataset to be downloaded.
        file_extension (str): The extension of the files to be processed. Defaults to ".csv".
    Returns:
        list[Path]: A list of Paths to the saved files.
    """
    paths = get_project_paths()
    raw_dir = paths['RAW']
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{slug}'...")
    dataset_path = kagglehub.dataset_download(slug)
    dataset_path = Path(dataset_path)

    saved_files = []

    for file in dataset_path.glob(f"*{file_extension}"):
        dest_file = raw_dir / file.name
        shutil.copy(file, dest_file)
        saved_files.append(dest_file)
        print(f"File saved: {dest_file}")

        try:
            df = pd.read_csv(dest_file)
            describe_dataset(df, name=file.stem)
            save_data(df, save_dir=raw_dir, base_filename=file.stem)
        except Exception as e:
            print(f"Error processing file '{file.name}': {e}")

    return saved_files

if __name__ == "__main__":
    download_dataset("samayashar/fraud-detection-transactions-dataset", file_extension=".csv")