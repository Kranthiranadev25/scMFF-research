import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import (
    LabelEncoder,
    label_binarize
)

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from lightgbm import LGBMClassifier

# --------------------------------------------
# LOAD DATA
# --------------------------------------------

X = pd.read_csv(
    'Example_data/Fusion_sum-expr-scpsd-scgnn-pca.txt.gz',
    sep=' ',
    header=None
)

y = pd.read_csv(
    'Example_data/cluster.csv',
    header=None
).values.ravel()

# --------------------------------------------
# LABEL ENCODING
# --------------------------------------------

le = LabelEncoder()

y = le.fit_transform(y)

classes = np.unique(y)

# --------------------------------------------
# SPLIT
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------
# MODEL
# --------------------------------------------

model = LGBMClassifier()

model.fit(X_train, y_train)

# --------------------------------------------
# ROC-AUC
# --------------------------------------------

y_score = model.predict_proba(X_test)

y_test_bin = label_binarize(
    y_test,
    classes=classes
)

plt.figure()

for i in range(len(classes)):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f'Class {i} AUC={roc_auc:.2f}'
    )

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig("plots/roc_curve.png")

plt.close()

# --------------------------------------------
# PR-AUC
# --------------------------------------------

plt.figure()

for i in range(len(classes)):

    precision, recall, _ = precision_recall_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )

    pr_auc = average_precision_score(
        y_test_bin[:, i],
        y_score[:, i]
    )

    plt.plot(
        recall,
        precision,
        label=f'Class {i} AP={pr_auc:.2f}'
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")

plt.legend()

plt.savefig("plots/pr_curve.png")

plt.close()

# --------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------

importance = model.feature_importances_

indices = np.argsort(importance)[-20:]

plt.figure(figsize=(10,6))

plt.barh(
    range(len(indices)),
    importance[indices]
)

plt.yticks(
    range(len(indices)),
    indices
)

plt.title("Feature Importance")

plt.savefig("plots/feature_importance.png")

plt.close()

# --------------------------------------------
# K-FOLD CV
# --------------------------------------------

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold
)

print("\nK-Fold CV Scores:")
print(scores)

print("\nMean CV Score:")
print(scores.mean())

# --------------------------------------------
# BOXPLOT
# --------------------------------------------

plt.figure()

plt.boxplot(scores)

plt.title("Cross Validation Score Distribution")

plt.savefig("plots/cv_boxplot.png")

plt.close()

print("\nAdvanced plots generated successfully.")