import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef
)

from lightgbm import LGBMClassifier

# ----------------------------------------
# LOAD LABELS
# ----------------------------------------

y = pd.read_csv(
    'Example_data/cluster.csv',
    header=None
).values.ravel()

le = LabelEncoder()

y = le.fit_transform(y)

# ----------------------------------------
# FEATURE FILES
# ----------------------------------------

feature_sets = {

    "expr_2000":
        'Example_data/expr_2000.csv',

    "scPSD":
        'Example_data/scPSD_embedding.txt.gz',

    "PCA":
        'Example_data/PCA_embedding.txt.gz',

    "scGNN":
        'Example_data/scGNN_embedding.csv',

    "Fusion_sum":
        'Example_data/Fusion_sum-expr-scpsd-scgnn-pca.txt.gz',

    "Adaptive_Fusion":
        'Example_data/adaptive_fusion.csv'
}

# ----------------------------------------
# RESULTS
# ----------------------------------------

results = []

# ----------------------------------------
# LOOP THROUGH FEATURES
# ----------------------------------------

for name, path in feature_sets.items():

    print(f"\nRunning: {name}")

    # Load feature data
    if path.endswith('.csv'):

        X = pd.read_csv(path, index_col=0)

        # Transpose if genes x cells
        if X.shape[0] < X.shape[1]:
            X = X.T

    else:

        X = pd.read_csv(
            path,
            header=None,
            sep=' '
        )

    # Convert columns to string
    X.columns = X.columns.astype(str)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Model
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
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

    # Save results
    results.append([
        name,
        accuracy,
        f1,
        mcc
    ])

# ----------------------------------------
# RESULTS TABLE
# ----------------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        'Feature_Set',
        'Accuracy',
        'F1_Score',
        'MCC'
    ]
)

print("\\nAblation Study Results")
print("------------------------")

print(results_df)

# Save
results_df.to_csv(
    'ablation_results.csv',
    index=False
)