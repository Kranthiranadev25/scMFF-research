import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average='weighted', zero_division=1),
    "recall": make_scorer(recall_score, average='weighted', zero_division=1),
    "f1": make_scorer(f1_score, average='weighted', zero_division=1),
    "mcc": make_scorer(matthews_corrcoef)
}

def cross_validation_model(X, y, model):
    try:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        scores = {
            "Accuracy": np.mean(cv_results['test_accuracy']),
            "Precision": np.mean(cv_results['test_precision']),
            "Recall": np.mean(cv_results['test_recall']),
            "F1": np.mean(cv_results['test_f1']),
            "MCC": np.mean(cv_results['test_mcc'])
        }
        return scores
    except Exception as e:
        print(f'Error in cross-validation: {e}')
        return {metric: np.nan for metric in scoring.keys()}
