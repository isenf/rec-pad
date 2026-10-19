# %%

import pandas as pd
import numpy as np
from rec_pad.bayes import *
import matplotlib.pyplot as plt
import seaborn as sns
from rec_pad.clustering import *
from rec_pad.features import *

# %%

rng = np.random.default_rng(23)
n=200

toy_df = pd.DataFrame({
    "length": np.concatenate([rng.lognormal(7.6, 0.5, n),
                              rng.lognormal(6.6, 0.6, n)]).round(0),
    "gc_content": np.concatenate([rng.normal(0.5, 0.05, n),
                                  rng.normal(0.42, 0.06, n)]).clip(0, 1).round(3),
    "rna_class": ["mRNA"]*int(n*1.3) +["lncRNA"]*int(n*.7)
})
toy_df.head()
# %%

X, y = prepare_xy(toy_df, class_col="rna_class")
X_scaled = minmax_dataframe(X)
X_minmax = scale_dataframe(X)
y_enc = encode_labels(y)

# %%

n_clusters = 2

labels_euc, Z_euc = hierarchical_clustering(X_scaled, metric="euclidean", method="average")
print(evaluate_clustering(X_scaled, labels_euc, y_true=y_enc))

# %%
labels_euc_corr, Z_euc_corr = hierarchical_clustering(X_minmax, metric="correlation", method="average")
print(evaluate_clustering(X_minmax, labels_euc_corr, y_true=y_enc))

# %%
