import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef
)

from lightgbm import LGBMClassifier

# -----------------------------------
# LOAD DATA
# -----------------------------------

X = pd.read_csv(
    'Example_data/Fusion_sum-expr-scpsd-scgnn-pca.txt.gz',
    header=None,
    sep=' '
)

y = pd.read_csv(
    'Example_data/cluster.csv',
    header=None
).values.ravel()

# -----------------------------------
# LABEL ENCODING
# -----------------------------------

le = LabelEncoder()

y = le.fit_transform(y)

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------
# BASE MODEL
# -----------------------------------

model = LGBMClassifier()

# -----------------------------------
# HYPERPARAMETER SEARCH
# -----------------------------------

param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [5],
    'num_leaves': [31]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1
)

# -----------------------------------
# TRAIN
# -----------------------------------

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# -----------------------------------
# PREDICTION
# -----------------------------------

y_pred = best_model.predict(X_test)

# -----------------------------------
# METRICS
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)

# -----------------------------------
# OUTPUT
# -----------------------------------

print("\\nBest Parameters")
print("------------------")
print(grid_search.best_params_)

print("\\nImproved Results")
print("------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"MCC      : {mcc:.4f}")