import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing_feature_engineering import load_and_transform

# 0. Paths
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"

MODEL_PATH  = RESULTS_DIR / "best_model.pkl"
TEST_FILE   = DATA_DIR    / "test.csv"
OUT_CSV     = RESULTS_DIR / "test_predictions.csv"


# 1. Load model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}.\n"
        "Run scripts/model_selection.py first."
    )

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model      = bundle["model"]
best_name  = bundle["best_name"]
print(f"Loaded model : {best_name}")


# 2. Load & engineer test features
if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found at {TEST_FILE}.\n"
        "Make sure test.csv is placed in the data/ directory."
    )

# test.csv may or may not contain Cover_Type
import pandas as _pd
_raw = _pd.read_csv(TEST_FILE)
has_target = "Cover_Type" in _raw.columns

X_test, y_test = load_and_transform(TEST_FILE, has_target=has_target)
print(f"Test set shape : {X_test.shape}")


# 3. Predict
y_pred = model.predict(X_test)

if has_target and y_test is not None:
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy (test.csv) : {acc:.4f}")
else:
    print("\nNo ground-truth labels in test.csv — predictions saved without accuracy.")


# 4. Save predictions
out_df = pd.DataFrame({"Cover_Type": y_pred})
out_df.index.name = "Id"
out_df.to_csv(OUT_CSV)
print(f"\nSaved in {OUT_CSV}")

print("\n predict.py complete.")