# %%

import os
import re
import pandas as pd
import seaborn as sns
from rec_pad.utils.proba import *
from rec_pad.utils.pdf_posterior import *
from rec_pad.utils.plot import *
from rec_pad.eda.io import *
from rec_pad.eda.features import *
from rec_pad.eda.plots import *
# from rec_pad.eda

# %%

path = "../../data/raw"
path_output = "../../output"
regex = r"\.0$"
class_col = "CLASS"

# %%
data, species = load_species(path=path)

# %%
# feature selection

for specie in species:
    data[specie] = filter_columns(
        df=data[specie],
        regex=regex,
        class_col=class_col
    )

# %%

for specie in species:
    ax = corr_heatmap(
        df=data[specie],
        class_col=class_col,
        annot=True,
        figsize=(20, 16)
    )
    save_fig(
        ax=ax, 
        path=f"{path_output}/raw/corr", 
        file_name=f"{specie}.png")
    plt.close(ax.figure)

# %%

data, votes = select_features(
    data,     
    threshold=0.8,
    class_col=class_col,
    min_votes=len(data)-1,
    return_votes=True
    )

# %%

