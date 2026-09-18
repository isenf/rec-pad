# %%

import os
import re
import pandas as pd
from utils.proba import prior_proba

# %%

path = "data/raw"
files = sorted(os.listdir(path=path))
species = [file[:-4] for file in files]
data = {}

# %%

for specie, file in zip(species, files):
    data[specie] = pd.read_csv(f"{path}/{file}", index_col="Unnamed: 0")

# %%

for specie in species:
    print(f"{specie} -> {data[specie].shape}")

# %%
# feature selection

column_names = data[species[0]].columns

# regex = r"\.0$|DELTA[1-5]$|^[^.]*$"
regex = r"\.0$"
pattern = re.compile(regex)

# for column_name in column_names:
filtered_cols = [c for c in column_names if re.search(pattern=pattern, string=c)]
filtered_cols.append("CLASS")
# %%

data_filtered = {}

for specie in species:
    data_filtered[specie] = data[specie][filtered_cols]
    # print(data_filtered[specie].columns)

# %%

for specie in species:
    print(specie, prior_proba(data_filtered[specie], "CLASS"), sep="\n", end="\n\n")

# %%
