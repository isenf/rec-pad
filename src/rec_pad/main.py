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
# from rec_pad.eda

# %%

path = "../../data/raw"
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

cols_non_red = []

for specie in species:
    cols_non_red.append(get_non_redundant(
        df=data[specie],
        threshold=0.8,
        class_col=class_col
    ))

