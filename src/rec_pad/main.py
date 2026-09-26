# %%

# import os
# import re
# import pandas as pd
# import seaborn as sns
from rec_pad.utils import *
from rec_pad.eda import * 

# %%

path = "../../data/raw"
path_output = "../../output"
regex = r"\.0$"
class_col = "CLASS"

# %%
data, species = io.load_species(path=path)

# %%
# data basic infos
infos = summarize_datasets(data=data, class_col=class_col)
print(infos)

# %%

for specie in species:
    print(f"\nspecie: {specie}\nmissings: {missing_report(data[specie])}\n"
          f"duplicate values: {duplicate_count(data[specie])}")

# %%
# drop duplicate values

for specie in species:
    data[specie] = drop_duplicates(data[specie])
    # print(f"duplicate values: {duplicate_count(data[specie])}")

# %%
# feature selection

for specie in species:
    data[specie] = features.filter_columns(
        df=data[specie],
        regex=regex,
        class_col=class_col
    )

# %%

for specie in species:
    ax = plots.corr_heatmap(
        df=data[specie],
        class_col=class_col,
        annot=True,
        figsize=(20, 16)
    )
    io.save_fig(
        fig=ax, 
        path=f"{path_output}/raw/corr", 
        file_name=f"{specie}.png")
    plt.close(ax.figure)

# %%

data, votes = features.select_features(
    data,     
    threshold=0.8,
    class_col=class_col,
    min_votes=len(data)-1,
    return_votes=True
    )

# %%
# dataset infos after feature selection

infos = stats.summarize_datasets(data=data, class_col=class_col)
print(infos)

# %%

# plots after processed data
for specie in species:
    fig = plots.violin_grid(
        df=data[specie], 
        features=[c for c in data[specie].columns if c!= class_col],
        class_col=class_col, 
        ncols=4,
        sharey=True)
    io.save_fig(
        fig=fig, 
        path=f"{path_output}/processed/violin",
        file_name=f"{specie}_normalized.png")

# %%

for specie in species:
    fig = plots.pair_plot(
        df=data[specie],
        class_col=class_col,
        features=[c for c in data[specie].columns if c!= class_col],
        title=f"Pair plot {specie}"
    )

# %%
