import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# LOAD RESULTS
# -----------------------------------

df = pd.read_csv('ablation_results.csv')

# Filter only fusion methods
fusion_df = df[
    df['Feature_Set'].isin([
        'Fusion_sum',
        'Adaptive_Fusion'
    ])
]

print(fusion_df)

# -----------------------------------
# ACCURACY COMPARISON
# -----------------------------------

plt.figure(figsize=(8,6))

plt.bar(
    fusion_df['Feature_Set'],
    fusion_df['Accuracy']
)

plt.ylabel("Accuracy")

plt.title(
    "Original Fusion vs Adaptive Fusion"
)

plt.tight_layout()

plt.savefig("fusion_comparison.png")

plt.show()