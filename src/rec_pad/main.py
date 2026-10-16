# %%

from rec_pad import (
    bayes,
    features,
    io,
    plots,
    proba,
    stats,
    helpers, 
    clustering
)

from pathlib import Path

# %%

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH = PROJECT_ROOT/"data"/"raw"
PATH_OUTPUT = PROJECT_ROOT/"output"
REGEX = r"\.0$"
CLASS_COL = "CLASS"

# %%
data, species = io.load_species(path=PATH)

# %%
# data basic infos
infos = stats.summarize_datasets(data=data, class_col=CLASS_COL)
print(infos)

# %%

for specie in species:
    print(f"\nspecie: {specie}\nmissings: {stats.missing_report(data[specie])}\n"
          f"duplicate values: {stats.duplicate_count(data[specie])}")

# %%
# drop duplicate values

for specie in species:
    data[specie] = stats.drop_duplicates(data[specie])
    # print(f"duplicate values: {duplicate_count(data[specie])}")

# %%
# feature selection

for specie in species:
    data[specie] = features.filter_columns(
        df=data[specie],
        regex=REGEX,
        class_col=CLASS_COL
    )

# %%

for specie in species:
    ax = plots.corr_heatmap(
        df=data[specie],
        class_col=CLASS_COL,
        annot=True,
        figsize=(20, 16)
    )
    helpers.save_and_close(
        fig=ax.figure, 
        subdir="/raw/corr", 
        file_name=f"{specie}.png")

# %%

data, votes = features.select_features(
    data,     
    threshold=0.8,
    class_col=CLASS_COL,
    min_votes=len(data)-1,
    return_votes=True
    )

print(f"votes:\n{votes}")

# %%
# dataset infos after feature selection

infos = stats.summarize_datasets(data=data, class_col=CLASS_COL)
print(infos)

# %%
stats.datasets_desc(data=data)

# %%

# plots after processed data
for specie in species:
    fig = plots.violin_grid(
        df=data[specie], 
        features=[c for c in data[specie].columns if c!= CLASS_COL],
        class_col=CLASS_COL, 
        ncols=4,
        sharey=True)
    helpers.save_and_close(
        fig=fig.figure, 
        subdir="processed/violin",
        file_name=f"{specie}_normalized.png")

# %%

for specie in species:
    fig = plots.pair_plot(
        df=data[specie],
        class_col=CLASS_COL,
        features=[c for c in data[specie].columns if c!= CLASS_COL],
        title=f"Pair plot {specie}"
    )
    helpers.save_and_close(
        fig=fig.figure, 
        subdir="processed/pairplot",
        file_name=f"{specie}.png")

# %%

for specie in species:
    fig = plots.parallel_coords(
        df=data[specie],
        class_col=CLASS_COL,
        normalize=True,
        figsize=(12, 6)
    )
    helpers.save_and_close(
        fig=fig.figure, 
        subdir="processed/parallel_coords",
        file_name=f"{specie}.png")
    

# %% 

# prior probability
helpers.print_proba_by_species(
    data=data,
    species=species,
    title="Prior Probability",
    func=proba.prior_proba,
    col=CLASS_COL
    )

# %%

# events probability
# event 1
event_1 = "`STRG.0` < 50.0"

helpers.print_proba_by_species(
    data=data,
    species=species,
    title=f"Event probability - {event_1}",
    func=proba.event_proba,
    event=event_1
)

# %%
# event 2
event_2 = "`T030C.0` < 400"

helpers.print_proba_by_species(
    data=data,
    species=species,
    title=f"Event probability - {event_2}",
    func=proba.event_proba,
    event=event_2
)

# %%
# event 3:
event_3 = "`ASPL.0` >= 1.5"
helpers.print_proba_by_species(
    data=data,
    species=species,
    title=f"Event probability - {event_3}",
    func=proba.event_proba,
    event=event_3
)

# %%

# union probability
# event 1 or event 2
helpers.print_proba_by_species(
    data=data,
    species=species,
    title=f"Union probability - {event_1} or {event_2}",
    func=proba.union_proba,
    cond1=event_1,
    cond2=event_2
)

# %%
# event 1 or event 3
helpers.print_proba_by_species(
    data=data,
    species=species,
    title=f"Union probability - {event_1} or {event_3}",
    func=proba.union_proba,
    cond1=event_1,
    cond2=event_3
)

# %%

# intersection probability
# event 1 and event 2
helpers.print_proba_by_species(
    data=data,
    species=species,
    title=f"Intersection probability - {event_1} and {event_2}",
    func=proba.intersection_proba,
    cond1=event_1,
    cond2=event_2
)

# %%
# event 1 and event 3
helpers.print_proba_by_species(
    data=data,
    species=species,
    title=f"Intersection probability - {event_1} and {event_3}",
    func=proba.intersection_proba,
    cond1=event_1,
    cond2=event_3
)

# %%

# conditional probability
# P(`STRG.0` < 50.0 | `CLASS` == 'mRNA')
cond1 = event_1
cond2 = "`CLASS` == 'mRNA'"
helpers.print_proba_by_species(
    data=data,
    species=species,
    title=f"Conditional probability - P({cond1} | {cond2})",
    func=proba.conditional_proba,
    cond1=cond1,
    cond2=cond2
)

# %%

cond1 = event_3
cond2 = "`CLASS` == 'lncRNA'"
helpers.print_proba_by_species(
    data=data,
    species=species,
    title=f"Conditional probability - P({cond1} | {cond2})",
    func=proba.conditional_proba,
    cond1=cond1,
    cond2=cond2
)

# %%

cond1 = event_1
cond2 = event_2
helpers.print_proba_by_species(
    data=data,
    species=species,
    title=f"Conditional probability - P({cond1} | {cond2})",
    func=proba.conditional_proba,
    cond1=cond1,
    cond2=cond2
)

# %%
# pdf and posterior probability
FEATURES = ["T030C.0", "ASS.0", "ASPL.0"]

for specie in species:
    for feat in FEATURES:
        res = bayes.pdf_posterior(data[specie], feature=feat, 
                            class_col=CLASS_COL, n_bins=16)

        # pdf curve
        fig_pdf = plots.plot_pdf(res, feature=feat, step=True,
                                title=f"Class-Conditional PDF - {specie} {feat}")
        helpers.save_and_close(
            fig=fig_pdf,
            subdir="pdf",
            file_name=f"{specie}_{feat}.jpg",
        )

        # posterior plots
        fig_post = plots.plot_binned_curves(res, "posterior", feature=feat, 
                                           title=f"Posterior probability - {specie} {feat}")
        helpers.save_and_close(
            fig=fig_post.figure,
            subdir="posterior",
            file_name=f"{specie}_{feat}.jpg",
        )

        # posterior grid plot
        fig_grid = plots.plot_posterior(res, feature=feat)
        helpers.save_and_close(
            fig=fig_grid,
            subdir="posterior_grid",
            file_name=f"{specie}_{feat}.jpg",
        )

# %% 
# clustering

from sklearn.metrics.cluster import contingency_matrix
import numpy as np

def purity_score(
    y_true,
    y_pred
):
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0))/np.sum(cm)

# %%
# average linkage - euclidean distance

data_minmax = {}
data_scaled = {}


for specie in species:
    data_minmax[specie] = features.minmax_dataframe(df=data[specie])
    data_scaled[specie] = features.scale_dataframe(df=data[specie])

# %%

from sklearn.preprocessing import LabelEncoder
X = data_minmax[species[0]].drop(columns=CLASS_COL)
y =  data_minmax[species[0]][CLASS_COL]

enc = LabelEncoder()
y_enc = enc.fit_transform(y)

# %%
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
euc_clustering = linkage(X, method="average", metric="euclidean")

# %%

n = 2
labels_euc = fcluster(euc_clustering, t=n, criterion="maxclust")

# %%

purity_euc = purity_score(y_true=y_enc, y_pred=labels_euc)
print(purity_euc)

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
dendrogram(euc_clustering, p=10)
# %%
# average linkage - correlation 



