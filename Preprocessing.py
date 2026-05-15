import pandas as pd
import numpy as np

def preprocess_RNASeq(df):

    # Convert to float
    df = df.astype(np.float32)

    # Log transform
    df = np.log1p(df)

    # Compute variance for each gene
    gene_variance = df.var(axis=1)

    # Select top 2000 highly variable genes
    top_genes = gene_variance.sort_values(ascending=False).head(2000).index

    # Subset dataframe
    df0 = df.loc[top_genes]

    return df0

