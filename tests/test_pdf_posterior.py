# %%

import pandas as pd
import numpy as np
from rec_pad.utils.pdf_posterior import *
import matplotlib.pyplot as plt
import seaborn as sns

# %%

toy_df0 = pd.DataFrame({
    "length": [1200, 2400, 3100, 800,
               600,950, 1850,220],
    "gc_content": [.48, .52, .41, .45, .38, .44, .36, .4],
    "rna_class": ["mRNA", "mRNA", "mRNA", "mRNA",
                  "lncRNA", "lncRNA", "lncRNA", "lncRNA"
                 ],
})
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

res0 = pdf_posterior(toy_df0, feature="gc_content", class_col="rna_class", n_bins=20)
res0

# %%

res = pdf_posterior(toy_df, feature="gc_content", class_col="rna_class", n_bins=20)
res
# %%

# sns.kdeplot(data=toy_df, x="gc_content", hue="rna_class", 
#             common_norm=False, # for area under curve sum 1 per curve
#             bw_adjust=.5)
# %%
# sns.kdeplot(data=toy_df, x="gc_content", hue="rna_class", 
#             common_norm=True, # for area under curve sum 1
#             bw_adjust=.5)
