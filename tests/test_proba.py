# %%

import pandas as pd
from rec_pad.utils.proba import (
    prior_proba, 
    event_proba,
    intersection_proba,
    union_proba,
    conditional_proba
)

# %%

toy_df = pd.DataFrame({
    "length": [1200, 2400, 3100, 800,
               600,950, 1850,220],
    "gc_content": [.48, .52, .41, .45, .38, .44, .36, .4],
    "rna_class": ["mRNA", "mRNA", "mRNA", "mRNA",
                  "lncRNA", "lncRNA", "lncRNA", "lncRNA"
                 ],
})

# %%
prior_proba(toy_df, "rna_class")

# %%
e2 = event_proba(toy_df, "length < 800")
e1 = event_proba(toy_df, "gc_content >= .5")
event_proba(toy_df, "gc_content > 1")

# %%

intersection_proba(toy_df, "length > 1000", "rna_class == 'mRNA'")

# %%
union_proba(toy_df, "length > 1000", "rna_class == 'mRNA'")
union_proba(toy_df, "gc_content < .5", "rna_class == 'lncRNA'")

# %%

conditional_proba(toy_df, "length > 1000", "rna_class == 'mRNA'")

