# Forest Cover Type Prediction

## Project Overview

This project builds a machine learning pipeline to classify forest cover types from
cartographic variables. Given a 30×30 meter land patch in the Roosevelt National Forest,
Colorado, the model predicts which of 7 tree species dominates it — using only terrain
measurements (elevation, slope, aspect, distances to water/roads/fire points, hillshade,
soil type, wilderness area). No satellite imagery is used.

The model assists environmental conservation agencies in forest management and
ecosystem strategy decisions.

---

## Results

| Metric | Value |
|---|---|
| Best model | Gradient Boosting |
| Train accuracy (Train set 1) | 0.9707 |
| Cross-val accuracy (5-fold) | 0.9319 |
| Test accuracy (Test set 1) | 0.9386 |
| **Final test accuracy (test.csv)** | **0.6935** |

> **Rule check:** Train accuracy 0.9707 < 0.98 

> **Rule check:** Final test accuracy 0.6935 > 0.65 

---

## How to Run from an Empty Environment

### 1. Clone the repository and enter the project folder
```bash
git clone https://learn.zone01oujda.ma/git/bdaanoun/forest-prediction.git
cd forest-prediction
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the data files
Make sure `train.csv`, `test.csv`, and `covtype.info` are in the `data/` folder.

### 5. (Optional) Run feature engineering standalone to verify setup
```bash
python scripts/preprocessing_feature_engineering.py
```

### 6. Run model selection — grid search over 5 models (takes time, leave overnight)
```bash
python scripts/model_selection.py
```
Outputs saved to `results/`: `best_model.pkl`, `confusion_matrix_heatmap.png`, `learning_curve_best_model.png`

### 7. Run predictions on the final test set
```bash
python scripts/predict.py
```
Outputs saved to `results/`: `test_predictions.csv`

---

## Python File Summaries

### `scripts/preprocessing_feature_engineering.py`
Defines two functions used by all other scripts:

- `engineer_features(df)` — takes a raw DataFrame and adds 10 derived columns (see Feature Engineering section below). Works on both train and test data.
- `load_and_transform(filepath, has_target)` — reads a CSV, applies `engineer_features`, and returns `(X, y)` for training data or `(X, None)` for test data without labels.

Can also be run standalone to verify the feature matrix shape and list all new columns.

### `scripts/model_selection.py`
The main training script. Imports `load_and_transform` from the preprocessing module, then:

1. Loads and engineers features from `train.csv`
2. Splits data into Train(1) 75% / Test(1) 25% using stratified split
3. Defines a grid search over 5 models: Logistic Regression, SVM, KNN, Random Forest, and Gradient Boosting — with `StandardScaler` inside a `Pipeline` for distance-based models
4. Runs `GridSearchCV` with 5-fold stratified cross-validation on Train(1)
5. Selects the best model by CV accuracy
6. Evaluates it on the held-out Test(1) and prints a full summary
7. Saves results to `results/`: confusion matrix heatmap, learning curve plot, and `best_model.pkl`

### `scripts/predict.py`
The last-day prediction script:

1. Loads `best_model.pkl` from `results/`
2. Applies the same feature engineering to `test.csv` via `load_and_transform`
3. Predicts `Cover_Type` for every row
4. If `test.csv` contains ground-truth labels, computes and prints final accuracy
5. Saves predictions to `results/test_predictions.csv`

---

## Feature Engineering

10 features are derived from the raw cartographic columns in `preprocessing_feature_engineering.py`.

| Feature | Formula | Reason |
|---|---|---|
| `Distance_To_Hydrology` | `sqrt(H_dist² + V_dist²)` | Euclidean distance to nearest water body — a tree model cannot compute a square root across two columns via splits |
| `Fire_Road_Diff` | `Dist_FirePoints − Dist_Roadways` | Captures the gap between fire exposure and road access — highlights remoteness vs. infrastructure |
| `Mean_Distance_To_Amenities` | `(H_water + H_roads + H_fire) / 3` | Average human infrastructure proximity — single summary of isolation level |
| `Elevation_Above_Water` | `Elevation − Vertical_Distance_To_Hydrology` | Soil moisture proxy — species distribution is strongly driven by how far above the water table a patch sits |
| `Mean_Hillshade` | `(Hillshade_9am + Noon + 3pm) / 3` | Average daily sun exposure across the three measurement times |
| `Hillshade_Range` | `max(9am, Noon, 3pm) − min(9am, Noon, 3pm)` | How much light varies through the day — proxy for slope steepness and aspect variability |
| `Cos_Aspect` | `cos(Aspect in radians)` | Aspect is circular (359° ≈ 1°). Raw degrees break near 0/360 — cosine puts it on a circle |
| `Sin_Aspect` | `sin(Aspect in radians)` | Required alongside Cos_Aspect to fully encode the circular angle without information loss |
| `Slope_Aspect_Interaction` | `Slope × Cos_Aspect` | A steep north-facing slope stays cool and moist; a steep south-facing slope dries out — one number captures both dimensions |
| `Soil_Type_Count` | `sum(Soil_Type_1 … Soil_Type_40)` | Number of active soil type bits per row — most rows = 1, but anomalies are ecologically meaningful |
| `Wilderness_Area_Count` | `sum(Wilderness_Area_1 … Wilderness_Area_4)` | Number of active wilderness area bits — flags unusual multi-area boundary patches |

---

## Model Selection Summary

Grid search with 5-fold stratified cross-validation. Data split: 75% Train(1) / 25% Test(1).
Distance-based models (LR, SVM, KNN) use `StandardScaler` inside a `Pipeline` so the scaler
only fits on training folds and never sees validation data. Tree-based models (RF, GB) do not
require scaling.

| Model | CV accuracy | Test(1) accuracy | Train accuracy |
|---|---|---|---|
| **Gradient Boosting** | **0.9319** | **0.9386** | **0.9707** |
| Random Forest | 0.9356 | 0.9434 | 0.9940 |
| KNN | 0.8914 | 0.9009 | 1.0000 |
| SVM | 0.8745 | 0.8757 | 0.8916 |
| Logistic Regression | 0.8104 | 0.8125 | 0.8123 |

**Gradient Boosting was selected** as the best model. Although Random Forest had a slightly
higher CV accuracy (0.9356 vs 0.9319), its train accuracy of 0.994 indicated overfitting.
Gradient Boosting's train accuracy of 0.9707 is healthy — close to its validation score —
making it the more trustworthy choice for unseen data.

---

## Project Structure

```
project/
│   README.md
│   requirements.txt
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── covtype.info
│
├── notebook/
│   └── EDA.ipynb
│
├── scripts/
│   ├── preprocessing_feature_engineering.py  — feature engineering, shared by all scripts
│   ├── model_selection.py                    — grid search, trains and saves best model
│   └── predict.py                            — loads model, predicts on test.csv
│
└── results/
    ├── confusion_matrix_heatmap.png
    ├── learning_curve_best_model.png
    ├── test_predictions.csv
    └── best_model.pkl
```