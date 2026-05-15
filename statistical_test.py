import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier

from scipy.stats import ttest_rel

# ----------------------------------------
# LOAD LABELS
# ----------------------------------------

y = pd.read_csv(
    'Example_data/cluster.csv',
    header=None
).values.ravel()

le = LabelEncoder()

y = le.fit_transform(y)

# ----------------------------------------
# LOAD FEATURES
# ----------------------------------------

fusion_sum = pd.read_csv(
    'Example_data/Fusion_sum-expr-scpsd-scgnn-pca.txt.gz',
    header=None,
    sep=' '
)

adaptive_fusion = pd.read_csv(
    'Example_data/adaptive_fusion.csv'
)

# Convert columns to string
fusion_sum.columns = fusion_sum.columns.astype(str)
adaptive_fusion.columns = adaptive_fusion.columns.astype(str)

# ----------------------------------------
# MODEL
# ----------------------------------------

model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)

# ----------------------------------------
# CROSS VALIDATION
# ----------------------------------------

scores_fusion = cross_val_score(
    model,
    fusion_sum,
    y,
    cv=5,
    scoring='accuracy'
)

scores_adaptive = cross_val_score(
    model,
    adaptive_fusion,
    y,
    cv=5,
    scoring='accuracy'
)

# ----------------------------------------
# PRINT SCORES
# ----------------------------------------

print("\\nFusion_sum Scores")
print(scores_fusion)

print("\\nAdaptive_Fusion Scores")
print(scores_adaptive)

# ----------------------------------------
# PAIRED T-TEST
# ----------------------------------------

t_stat, p_value = ttest_rel(
    scores_fusion,
    scores_adaptive
)

print("\\nStatistical Test")
print("-------------------")

print(f"T-statistic : {t_stat:.4f}")
print(f"P-value     : {p_value:.6f}")

# ----------------------------------------
# INTERPRETATION
# ----------------------------------------

alpha = 0.05

if p_value < alpha:
    print(
        "\\nResult: Improvement is statistically significant."
    )
else:
    print(
        "\\nResult: Improvement is NOT statistically significant."
    )
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier

from scipy.stats import ttest_rel

# ----------------------------------------
# LOAD LABELS
# ----------------------------------------

y = pd.read_csv(
    'Example_data/cluster.csv',
    header=None
).values.ravel()

le = LabelEncoder()

y = le.fit_transform(y)

# ----------------------------------------
# LOAD FEATURES
# ----------------------------------------

fusion_sum = pd.read_csv(
    'Example_data/Fusion_sum-expr-scpsd-scgnn-pca.txt.gz',
    header=None,
    sep=' '
)

adaptive_fusion = pd.read_csv(
    'Example_data/adaptive_fusion.csv'
)

# Convert columns to string
fusion_sum.columns = fusion_sum.columns.astype(str)
adaptive_fusion.columns = adaptive_fusion.columns.astype(str)

# ----------------------------------------
# MODEL
# ----------------------------------------

model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)

# ----------------------------------------
# CROSS VALIDATION
# ----------------------------------------

scores_fusion = cross_val_score(
    model,
    fusion_sum,
    y,
    cv=5,
    scoring='accuracy'
)

scores_adaptive = cross_val_score(
    model,
    adaptive_fusion,
    y,
    cv=5,
    scoring='accuracy'
)

# ----------------------------------------
# PRINT SCORES
# ----------------------------------------

print("\\nFusion_sum Scores")
print(scores_fusion)

print("\\nAdaptive_Fusion Scores")
print(scores_adaptive)

# ----------------------------------------
# PAIRED T-TEST
# ----------------------------------------

t_stat, p_value = ttest_rel(
    scores_fusion,
    scores_adaptive
)

print("\\nStatistical Test")
print("-------------------")

print(f"T-statistic : {t_stat:.4f}")
print(f"P-value     : {p_value:.6f}")

# ----------------------------------------
# INTERPRETATION
# ----------------------------------------

alpha = 0.05

if p_value < alpha:
    print(
        "\\nResult: Improvement is statistically significant."
    )
else:
    print(
        "\\nResult: Improvement is NOT statistically significant."
    )