import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# LOAD RESULTS
# -----------------------------------

df = pd.read_csv('ablation_results.csv')

print(df)

# -----------------------------------
# ACCURACY PLOT
# -----------------------------------

plt.figure(figsize=(10,6))

plt.bar(
    df['Feature_Set'],
    df['Accuracy']
)

plt.xlabel("Feature Set")
plt.ylabel("Accuracy")
plt.title("Ablation Study - Accuracy Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("ablation_accuracy.png")

plt.show()

# -----------------------------------
# F1 SCORE PLOT
# -----------------------------------

plt.figure(figsize=(10,6))

plt.bar(
    df['Feature_Set'],
    df['F1_Score']
)

plt.xlabel("Feature Set")
plt.ylabel("F1 Score")
plt.title("Ablation Study - F1 Score Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("ablation_f1.png")

plt.show()

# -----------------------------------
# MCC PLOT
# -----------------------------------

plt.figure(figsize=(10,6))

plt.bar(
    df['Feature_Set'],
    df['MCC']
)

plt.xlabel("Feature Set")
plt.ylabel("MCC")
plt.title("Ablation Study - MCC Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("ablation_mcc.png")

plt.show()