import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

# --------------------------------------
# LOAD DATA
# --------------------------------------

X = pd.read_csv(
    'Example_data/Fusion_sum-expr-scpsd-scgnn-pca.txt.gz',
    sep=' ',
    header=None
)

y = pd.read_csv(
    'Example_data/cluster.csv',
    header=None
).values.ravel()

le = LabelEncoder()

y = le.fit_transform(y)

# --------------------------------------
# SPLIT
# --------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------
# MODEL
# --------------------------------------

model = LGBMClassifier()

model.fit(X_train, y_train)

# --------------------------------------
# SHAP
# --------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)