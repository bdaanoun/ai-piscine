import numpy as np
import pandas as pd
from pathlib import Path



DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE  = DATA_DIR / "test.csv"


# 2. Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print(df.head())
    df = df.copy()

    df["Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    df["Elevation_Above_Water"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    df["Fire_Road_Diff"] = (
        df["Horizontal_Distance_To_Fire_Points"]
        - df["Horizontal_Distance_To_Roadways"]
    )

    df["Mean_Distance_To_Amenities"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3

    print(df.head())
    return df


# 3. Load & transform
def load_and_transform(filepath: Path, has_target: bool = True):
    df = pd.read_csv(filepath)
    df = engineer_features(df)

    if has_target:
        y = df["Cover_Type"]
        X = df.drop(columns=["Cover_Type"])
        return X, y
    else:
        return df, None


# 4. Main (optional stand-alone run)
if __name__ == "__main__":
    X, y = load_and_transform(TRAIN_FILE, has_target=True)

    print(f"Feature matrix shape : {X.shape}")
    print(f"Target distribution  :\n{y.value_counts().sort_index()}")
    print(f"\nEngineered columns added:")
    original_cols = pd.read_csv(TRAIN_FILE).drop(columns=["Cover_Type"]).columns
    new_cols = [c for c in X.columns if c not in original_cols]
    for c in new_cols:
        print(f"  + {c}")
