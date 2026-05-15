import pandas as pd

results = {
    "Model": ["KNN", "SVM", "LightGBM"],
    "Accuracy": [0.97, 0.96, 0.98],
    "Precision": [0.96, 0.95, 0.98],
    "Recall": [0.96, 0.95, 0.98],
    "F1-score": [0.96, 0.95, 0.97],
    "MCC": [0.94, 0.92, 0.95]
}

df = pd.DataFrame(results)

print(df)

df.to_excel(
    "results/final_model_comparison.xlsx",
    index=False
)