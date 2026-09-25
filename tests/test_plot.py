# %%

import pandas as pd
import numpy as np
from rec_pad.utils.plot import *
from rec_pad.utils.pdf_posterior import pdf_posterior
import matplotlib.pyplot as plt
import seaborn as sns

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
res = pdf_posterior(toy_df, "gc_content", "rna_class", n_bins=20)
plot_binned_curves(res, "likelihood", "gc_content", "rna_class")
# %%
plot_binned_curves(res, "joint", "gc_content", "rna_class")
# %%
plot_binned_curves(res, "posterior", "gc_content", "rna_class")

# %%
plot_binned_curves(res, "counts", "gc_content", "rna_class", marker="d")


# %%
plot_binned_curves(res, "evidence", "gc_content", "rna_class", marker="d")

# %%

plot_binned_hist(res, "counts")

# %%
plot_binned_hist(res, 'prior', bar_width=0.8, xlabel="class")

# %%
plot_pdf_posterior(res, feature="gc_content", )
