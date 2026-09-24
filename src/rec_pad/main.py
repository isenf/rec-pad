# %%

import os
import re
import pandas as pd
import seaborn as sns
from rec_pad.utils.proba import *
from rec_pad.utils.pdf_posterior import *
from rec_pad.utils.plot import *

# %%

path = "../../data/raw"
files = sorted(os.listdir(path=path))
species = [file[:-4] for file in files]
data = {}

# %%

for specie, file in zip(species, files):
    data[specie] = pd.read_csv(f"{path}/{file}")

# %%
# feature selection

column_names = data[species[0]].columns

# regex = r"\.0$|DELTA[1-5]$|^[^.]*$"
regex = r"\.0$" # just the original network metrics
pattern = re.compile(regex)

# for column_name in column_names:
filtered_cols = [c for c in column_names if re.search(pattern=pattern, string=c)]
filtered_cols.append("CLASS")

# %%

data_filtered = {}

for specie in species:
    data_filtered[specie] = data[specie][filtered_cols]

# %%

# select features non-redundant features
all_keep = {}
for specie in species:
    corr = data_filtered[specie].drop(columns="CLASS").corr().abs()
    corr_bool = (corr>.8)
    keep = []

    for col in corr_bool.columns:
        if not any(corr_bool.loc[col, k] for k in keep):
            keep.append(col)
    all_keep[specie] = keep

# %%

all_equal = all(
    all_keep[species[0]] == all_keep[specie]
    for specie in species
)

# %%
cols = set()
for specie in species:
    for col in all_keep[specie]:
        # print(col)
        cols.add(col)
cols
# %%

# selected_cols = all_keep[species[0]]
# selected_cols.append("CLASS")
all_cols = list(cols) + ["CLASS"]

for specie in species:
    data[specie] = data[specie][all_cols]

# %%
for specie in species:
    sns.pairplot(data[specie], hue="CLASS", )
    plt.savefig(f"../../output/pairplot/{specie}.png", dpi=400)
    plt.show()

# %%
for specie in species:
    print(f"{specie} -> {data[specie].shape}")

# %%

for feat in cols:
    sns.violinplot(data=data[species[0]], x=feat, hue="CLASS", split=True, gap=1.0, )
    plt.show()

# %%

from pandas.plotting import parallel_coordinates
df = data[species[-1]]
df = df[df.columns[:-1]]
# %%
df_n = (df - df.min()) / (df.max() - df.min())
df_n["CLASS"] = data[species[-1]]["CLASS"].values
 # %%

fig, ax = plt.subplots(figsize=(20, 8))
parallel_coordinates(df_n, ax=ax, class_column="CLASS", color=("blue", "orange"), lw=.5, alpha=.6)

