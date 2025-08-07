import sys
import pandas as pd
import numpy as np
from load import save_data
from pathlib import Path
from sklearn.impute import KNNImputer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.paths import get_project_paths

