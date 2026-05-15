import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
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

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# LOAD DATA
# -----------------------------

X = pd.read_csv(
    'Example_data/Fusion_sum-expr-scpsd-scgnn-pca.txt.gz',
    header=None,
    sep=' '
)

y = pd.read_csv(
    'Example_data/cluster.csv',
    header=None
).values.ravel()

# -----------------------------
# LABEL ENCODING
# -----------------------------

le = LabelEncoder()
y = le.fit_transform(y)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# MODEL
# -----------------------------

model = KNeighborsClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# METRICS
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
mcc = matthews_corrcoef(y_test, y_pred)

print("\\nEvaluation Metrics")
print("-------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"MCC      : {mcc:.4f}")

# -----------------------------
# CROSS VALIDATION
# -----------------------------

cv_scores = cross_val_score(model, X, y, cv=5)

print("\\nCross Validation Scores")
print("------------------------")
print(cv_scores)
print(f"Mean CV Score: {cv_scores.mean():.4f}")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# -----------------------------
# ROC CURVE
# -----------------------------

# Binarize labels
classes = np.unique(y)

y_test_bin = label_binarize(y_test, classes=classes)

# Predict probabilities
y_score = model.predict_proba(X_test)

# ROC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC
plt.figure()

for i in range(len(classes)):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f'Class {i} (AUC = {roc_auc[i]:.2f})'
    )

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig("roc_curve.png")

plt.show()

from sklearn.model_selection import learning_curve

# -----------------------------
# LEARNING CURVE
# -----------------------------

train_sizes, train_scores, test_scores = learning_curve(
    model,
    X,
    y,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure()

plt.plot(train_sizes, train_mean, label='Training Score')
plt.plot(train_sizes, test_mean, label='Validation Score')

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")

plt.legend()

plt.savefig("learning_curve.png")

plt.show()

from sklearn.manifold import TSNE

# -----------------------------
# t-SNE VISUALIZATION
# -----------------------------

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

X_tsne = tsne.fit_transform(X.iloc[:1000])

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y[:1000],
    s=10
)

plt.title("t-SNE Visualization")

plt.savefig("tsne_plot.png")

plt.show()