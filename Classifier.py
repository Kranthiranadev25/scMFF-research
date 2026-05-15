from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def get_model(name):
    try:
        return {
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "SVM": SVC(kernel='rbf', C=1, probability=True),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            "LightGBM": LGBMClassifier(),
            "NaiveBayes": GaussianNB(),
            "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        }[name]
    except KeyError:
        raise ValueError(f"Model '{name}' is not supported.")
