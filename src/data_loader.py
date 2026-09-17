
from pathlib import Path
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "water_potability.csv"

def load_dataset(path=DEFAULT_DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset is empty.")
    return df

def detect_target(df):
    candidates = ["Potability", "potability", "target", "Target", "label", "Label", "class", "Class"]
    for c in candidates:
        if c in df.columns:
            return c
    binary = [c for c in df.columns if df[c].dropna().nunique() == 2 and pd.api.types.is_numeric_dtype(df[c])]
    if len(binary) == 1:
        return binary[0]
    raise ValueError("Could not identify a unique binary target column.")

def dataset_summary(df, target):
    numeric = df.select_dtypes(include="number")
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "target": target,
        "target_unique_values": sorted(df[target].dropna().unique().tolist()),
        "class_distribution": {str(k): int(v) for k, v in df[target].value_counts().sort_index().items()},
        "descriptive_statistics": numeric.describe().round(4).to_dict(),
    }
