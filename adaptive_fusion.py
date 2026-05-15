import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------
# LOAD FEATURES
# ----------------------------------------

expr = pd.read_csv(
    'Example_data/expr_2000.csv',
    index_col=0
).T

scpsd = pd.read_csv(
    'Example_data/scPSD_embedding.txt.gz',
    header=None,
    sep=' '
)

pca = pd.read_csv(
    'Example_data/PCA_embedding.txt.gz',
    header=None,
    sep=' '
)

scgnn = pd.read_csv(
    'Example_data/scGNN_embedding.csv',
    index_col=0
)

# ----------------------------------------
# NORMALIZATION
# ----------------------------------------

scaler = MinMaxScaler()

expr = scaler.fit_transform(expr)
scpsd = scaler.fit_transform(scpsd)
pca = scaler.fit_transform(pca)
scgnn = scaler.fit_transform(scgnn)

# ----------------------------------------
# MATCH DIMENSIONS
# ----------------------------------------

min_dim = min(
    expr.shape[1],
    scpsd.shape[1],
    pca.shape[1],
    scgnn.shape[1]
)

expr = expr[:, :min_dim]
scpsd = scpsd[:, :min_dim]
pca = pca[:, :min_dim]
scgnn = scgnn[:, :min_dim]

# ----------------------------------------
# ADAPTIVE WEIGHTS
# ----------------------------------------

w_expr = 0.35
w_scpsd = 0.30
w_pca = 0.15
w_scgnn = 0.20

# ----------------------------------------
# FUSION
# ----------------------------------------

fusion = (
    w_expr * expr +
    w_scpsd * scpsd +
    w_pca * pca +
    w_scgnn * scgnn
)

# ----------------------------------------
# SAVE
# ----------------------------------------

fusion_df = pd.DataFrame(fusion)

fusion_df.to_csv(
    'Example_data/adaptive_fusion.csv',
    index=False
)

print("\\nAdaptive Fusion Completed")
print("----------------------------")
print(f"Fusion Shape: {fusion_df.shape}")