import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

X = pd.read_csv(
    'Example_data/Fusion_sum-expr-scpsd-scgnn-pca.txt.gz',
    sep=' ',
    header=None
)

y = pd.read_csv(
    'Example_data/cluster.csv',
    header=None
).values.ravel()

# ------------------------------------------------
# LABEL ENCODING
# ------------------------------------------------

le = LabelEncoder()
y = le.fit_transform(y)

# ------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# MODELS
# ------------------------------------------------

models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "LightGBM": LGBMClassifier()
}

results = []

# ------------------------------------------------
# TRAIN + EVALUATE
# ------------------------------------------------

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(
        y_test,
        y_pred,
        average='weighted'
    )

    rec = recall_score(
        y_test,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5
    )

    results.append([
        name,
        acc,
        prec,
        rec,
        f1,
        mcc,
        cv_scores.mean()
    ])

    # --------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.title(f"{name} Confusion Matrix")

    plt.savefig(f"plots/{name}_confusion_matrix.png")

    plt.close()

# ------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "CV Score"
    ]
)

print(results_df)

results_df.to_excel(
    "results/model_comparison.xlsx",
    index=False
)

print("\nResults saved successfully.")