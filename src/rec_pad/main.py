# %%

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
    io.save_fig(
        fig=fig, 
        path=f"{path_output}/processed/pairplot",
        file_name=f"{specie}.png")

# %%

for specie in species:
    fig = plots.parallel_coords(
        df=data[specie],
        class_col=class_col,
        normalize=True,
        figsize=(12, 6)
    )
    io.save_fig(
        fig=fig, 
        path=f"{path_output}/processed/parallel_coords",
        file_name=f"{specie}.png")

# %% 

# prior probability
for specie in species:
    p_prior = proba.prior_proba(data[specie], col=class_col)
    print(f"\n{specie} - Prior Probability")
    for cls, p in p_prior.items():
        print(f"{cls}: {p:.4f}")

# %%

# events probability
# event 1
event_1 = "`STRG.0` < 50.0"
for specie in species:
    p_event = proba.event_proba(df=data[specie], event=event_1)
    print(f"\n{specie} - {event_1} Probability")
    for cls, p in p_event.items():
        print(f"{cls}: {p:.4f}")

# %%
# event 2
event_2 = "`T030C.0` < 400"
for specie in species:
    p_event = proba.event_proba(df=data[specie], event=event_2)
    print(f"\n{specie} - {event_2} Probability")
    for cls, p in p_event.items():
        print(f"{cls}: {p:.4f}")

# %%
# event 3:
event_3 = "`ASPL.0` >= 1.5"
for specie in species:
    p_event = proba.event_proba(df=data[specie], event=event_3)
    print(f"\n{specie} - {event_3} Probability")
    for cls, p in p_event.items():
        print(f"{cls}: {p:.4f}")

# %%

# union probability
# event 1 or event 2
for specie in species:
    p_union = proba.union_proba(df=data[specie], 
                                cond1=event_1,
                                cond2=event_2)
    print(f"\n{specie} - {event_1} or {event_2} Probability")
    for cls, p in p_union.items():
        print(f"{cls}: {p:.4f}")

# %%
# event 1 or event 3
for specie in species:
    p_union = proba.union_proba(df=data[specie], 
                                cond1=event_1,
                                cond2=event_3)
    print(f"\n{specie} - {event_1} or {event_3} Probability")
    for cls, p in p_union.items():
        print(f"{cls}: {p:.4f}")

# %%

# intersection probability
# event 1 and event 2
for specie in species:
    p_inter = proba.intersection_proba(df=data[specie], 
                                       cond1=event_1,
                                       cond2=event_2)
    print(f"\n{specie} - {event_1} and {event_2} Probability")
    for cls, p in p_inter.items():
        print(f"{cls}: {p:.4f}")

# %%
# event 1 and event 3
for specie in species:
    p_inter = proba.intersection_proba(df=data[specie], 
                                       cond1=event_1,
                                       cond2=event_3)
    print(f"\n{specie} - {event_1} and {event_3} Probability")
    for cls, p in p_inter.items():
        print(f"{cls}: {p:.4f}")

# %%

# conditional probability
# P(`STRG.0` < 50.0 | `CLASS` == 'mRNA')
cond_1 = event_1
cond_2 = "`CLASS` == 'mRNA'"
for specie in species:
    p_cond = proba.intersection_proba(df=data[specie], 
                                       cond1=cond_1,
                                       cond2=cond_2)
    print(f"\n{specie} - P({cond_1} | {cond_2})")
    for cls, p in p_cond.items():
        print(f"{cls}: {p:.4f}")

# %%

cond_1 = event_3
cond_2 = "`CLASS` == 'lncRNA'"
for specie in species:
    p_cond = proba.intersection_proba(df=data[specie], 
                                       cond1=cond_1,
                                       cond2=cond_2)
    print(f"\n{specie} - P({cond_1} | {cond_2})")
    for cls, p in p_cond.items():
        print(f"{cls}: {p:.4f}")

# %%

cond_1 = event_1
cond_2 = event_2
for specie in species:
    p_cond = proba.intersection_proba(df=data[specie], 
                                       cond1=cond_1,
                                       cond2=cond_2)
    print(f"\n{specie} - P({cond_1} | {cond_2})")
    for cls, p in p_cond.items():
        print(f"{cls}: {p:.4f}")

# %%

feats = ["T030C.0", "ASS.0", "ASPL.0"]

for specie in species:
    for feat in feats:
        res = pdf_posterior(data[specie], feature=feat, class_col=class_col, n_bins=16)
        fig = plot.plot_pdf_posterior(res, feat)
        io.save_fig(
            fig,
            path=f"{path_output}/posterior",
            file_name=f"{specie}_{feat}.png",
        )
