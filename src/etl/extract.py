import sys
import kagglehub
import shutil
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.paths import get_project_paths
from src.utils.dataset_describer import describe_dataset

def download_dataset(slug: str, file_extension: str = ".csv") -> list[Path]:
    """Downloads a dataset from Kaggle and saves it to the project's 'raw' directory.

    Args:
        slug (str): The slug of the dataset to download.
        file_extension (str, optional): The file extension to filter for. Defaults to ".csv".

    Returns:
        list[Path]: A list of the downloaded files. If no files are found, an empty list is returned.
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
        except Exception as e:
            print(f"Error reading file: {e}")

    return saved_files

if __name__ == "__main__":
    download_dataset("samayashar/fraud-detection-transactions-dataset", file_extension=".csv")