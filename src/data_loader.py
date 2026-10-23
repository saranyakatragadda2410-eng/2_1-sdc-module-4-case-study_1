import pandas as pd

def load_data(path: str = "data/house_data.csv") -> pd.DataFrame:
    """Load the house price dataset."""
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df