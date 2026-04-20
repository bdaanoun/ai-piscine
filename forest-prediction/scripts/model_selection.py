import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Local helper
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing_feature_engineering import load_and_transform

warnings.filterwarnings("ignore")

# 0. Paths
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_FILE  = DATA_DIR / "train.csv"


# 1. Load & split
print("Loading and engineering features …")
X, y = load_and_transform(TRAIN_FILE, has_target=True)

# Level-0 split: train(1) / held-out test(1)  ← <33 %
X_train, X_test1, y_train, y_test1 = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"  Train(1) : {X_train.shape}   Test(1) : {X_test1.shape}")

# Cross-validation strategy for grid search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# 2. Model zoo + hyper-parameter grids
search_space = {
    "Logistic Regression": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "param_grid": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "saga"],
        },
    },
    "SVM": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(random_state=42)),
        ]),
        "param_grid": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"],
        },
    },
    "KNN": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]),
        "param_grid": {
            "model__n_neighbors": [3, 5, 4], # 15, 21, 31
            "model__weights": ["uniform", "distance"],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [200, 300],
            "max_depth": [15, 20],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [5, 10],
        },
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 4],
            "subsample": [0.8],
            "min_samples_leaf": [5, 10],
        },
    },
}


# 3. Grid search
results = {}

for name, cfg in search_space.items():
    print(f"\n{'─'*50}")
    print(f"Grid-searching: {name} …")

    gs = GridSearchCV(
        estimator  = cfg["estimator"],
        param_grid = cfg["param_grid"],
        cv         = cv,
        scoring    = "accuracy",
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,
    )
    gs.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, gs.predict(X_train))
    val_acc   = gs.best_score_
    test_acc  = accuracy_score(y_test1, gs.predict(X_test1))

    results[name] = {
        "gs":        gs,
        "best_params": gs.best_params_,
        "train_acc": train_acc,
        "val_acc":   val_acc,
        "test_acc":  test_acc,
    }

    print(f"  Best params : {gs.best_params_}")
    print(f"  Train acc   : {train_acc:.4f}")
    print(f"  Val acc (CV): {val_acc:.4f}")
    print(f"  Test(1) acc : {test_acc:.4f}")


# 4. Select best model (by CV val accuracy)
summary = pd.DataFrame(
    {n: {"train_acc": v["train_acc"],
         "val_acc":   v["val_acc"],
         "test_acc":  v["test_acc"]}
     for n, v in results.items()}
).T.sort_values("val_acc", ascending=False)

print("\n\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print(summary.to_string())

best_name   = summary.index[0]
best_result = results[best_name]
best_model  = best_result["gs"].best_estimator_

print(f"\n Best model : {best_name}")
print(f"   Train acc   : {best_result['train_acc']:.4f}  (must be < 0.98)")
print(f"   Val acc(CV) : {best_result['val_acc']:.4f}")
print(f"   Test(1) acc : {best_result['test_acc']:.4f}")

if best_result["train_acc"] >= 0.98:
    print(f"\n Warning: Train accuracy {best_result['train_acc']:.4f} ≥ 0.98 — model may be overfit.")
    print("   Consider tightening max_depth or increasing min_samples_leaf.")
else:
    print(f"\n Train accuracy {best_result['train_acc']:.4f} < 0.98 — looks healthy.")


# 5. Confusion matrix  : DataFrame + heatmap
class_labels = sorted(y.unique())
y_pred_test  = best_model.predict(X_test1)
cm           = confusion_matrix(y_test1, y_pred_test, labels=class_labels)

cm_df = pd.DataFrame(
    cm,
    index   = [f"True_{c}"      for c in class_labels],
    columns = [f"Predicted_{c}" for c in class_labels],
)

print("\nConfusion Matrix (Test set 1):")
print(cm_df.to_string())

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm_df, annot=True, fmt="d", cmap="Blues",
    linewidths=0.5, ax=ax
)
ax.set_title(f"Confusion Matrix — {best_name}\n(Test set 1)", fontsize=14)
ax.set_ylabel("True label",      fontsize=12)
ax.set_xlabel("Predicted label", fontsize=12)
plt.tight_layout()

cm_path = RESULTS_DIR / "confusion_matrix_heatmap.png"
fig.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"\nSaved → {cm_path}")


# 6. Learning curve
print("\nComputing learning curve (this may take a moment) …")

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X_train, y_train,
    cv            = StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    train_sizes   = np.linspace(0.1, 1.0, 8),
    scoring       = "accuracy",
    n_jobs        = -1,
    shuffle       = True,
    random_state  = 42,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(train_sizes, train_mean, "o-", color="royalblue",  label="Training score")
ax.plot(train_sizes, val_mean,   "o-", color="darkorange", label="Cross-val score")
ax.fill_between(train_sizes,
                train_mean - train_std, train_mean + train_std,
                alpha=0.15, color="royalblue")
ax.fill_between(train_sizes,
                val_mean - val_std, val_mean + val_std,
                alpha=0.15, color="darkorange")
ax.set_title(f"Learning Curve — {best_name}", fontsize=14)
ax.set_xlabel("Training set size", fontsize=12)
ax.set_ylabel("Accuracy",          fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

lc_path = RESULTS_DIR / "learning_curve_best_model.png"
fig.savefig(lc_path, dpi=150)
plt.close(fig)
print(f"Saved → {lc_path}")


# 7. Save model
model_path = RESULTS_DIR / "best_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": best_model, "best_name": best_name}, f)
print(f"Saved → {model_path}")

print("\n  model_selection.py complete.")