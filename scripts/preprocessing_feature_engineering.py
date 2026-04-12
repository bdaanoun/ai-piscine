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

    # Euclidean distance to nearest water body
    df["Distance_To_Hydrology"] = np.sqrt(df["Horizontal_Distance_To_Hydrology"] ** 2 + df["Vertical_Distance_To_Hydrology"] ** 2)

    # Proximity gap: fire ignition points vs road access
    df["Fire_Road_Diff"] = (df["Horizontal_Distance_To_Fire_Points"] - df["Horizontal_Distance_To_Roadways"])

    # Mean distance to human/infra features
    df["Mean_Distance_To_Amenities"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]) / 3

    # Elevation relative to water (positive = above water)
    df["Elevation_Above_Water"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # Average of morning + noon + afternoon shade
    df["Mean_Hillshade"] = (
        df["Hillshade_9am"]
        + df["Hillshade_Noon"]
        + df["Hillshade_3pm"]
    ) / 3

    # Hillshade range: proxy for slope/aspect variability
    df["Hillshade_Range"] = (
        df[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]].max(axis=1)
        - df[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]].min(axis=1)
    )

    # Cosine of aspect in radians (north-facing tendency)
    df["Cos_Aspect"] = np.cos(np.radians(df["Aspect"]))
    df["Sin_Aspect"] = np.sin(np.radians(df["Aspect"]))

    # Slope × aspect interaction
    df["Slope_Aspect_Interaction"] = df["Slope"] * df["Cos_Aspect"]

    # --- Soil-type aggregate
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]
    df["Soil_Type_Count"] = df[soil_cols].sum(axis=1)

    # --- Wilderness-area aggregate ---
    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    df["Wilderness_Area_Count"] = df[wild_cols].sum(axis=1)

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
