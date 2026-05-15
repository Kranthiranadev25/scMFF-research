import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

# ------------------------------------
# LOAD DATA
# ------------------------------------

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

# ------------------------------------
# SPLIT
# ------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------
# MODEL
# ------------------------------------

model = LGBMClassifier()

params = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100]
}

grid = GridSearchCV(
    model,
    params,
    cv=3,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest Score:")
print(grid.best_score_)