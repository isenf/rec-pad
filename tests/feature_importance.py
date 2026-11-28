# %%

from rec_pad import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%

PATH = "../output/classification"
MODELS = ["GradientBoosting", "RandomForest"]
TAGS = ["all", "selected"]
SPECIES = ["Danio_rerio", "Mus_musculus", "Xenopus_tropicalis"]

# %%

def get_feature_importance(
    path: str,
    model_name: str,
    tag: str,
    specie:str
):
    artifact = io.load_classifier(f"{path}/{model_name}/{tag}/{specie}.sav")
    pipe = artifact["model"]
    model = pipe.named_steps['model']

    return pd.Series(
        model.feature_importances_,
        index=pipe.feature_names_in_,
        name=f"{model_name}_{tag}_{specie}"
    )

# %%

def heatmap_model(
    path=PATH,
    model=MODELS[0],
    tag=TAGS[0],
    species=SPECIES,
    top_n:int=10,
    figsize=(12, 6),
    cmap:str="Blues",
    save:bool=True, 
    annot_size: int=8
): 
    rows = {}
    for specie in species:
        rows[specie] = get_feature_importance(path, model, tag, specie)
    df = pd.DataFrame(rows).T.fillna(0.0)
    order = df.mean(axis=0).sort_values(ascending=False).index[:top_n]
    df = df[order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df, cmap=cmap, annot=True, fmt=".4f", annot_kws={"size": annot_size},
        cbar_kws={"label": "Feature Importance"}, ax=ax, vmin=0.0, vmax=1.0
    )
    ax.set_title(f"Feature importance - {model} ({tag})")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Species")
    plt.tight_layout()
    if save:
        out = f"{path}/feature_importance"
        io._make_dir(out)
        plt.savefig(f"{out}/{model}_{tag}.jpg", dpi=250)
heatmap_model()

# %%

if __name__ == '__main__':
    for model in MODELS:
        for tag in TAGS:
            heatmap_model(model=model, tag=tag, top_n=5, figsize=(8, 6), 
                          annot_size=12
                          )
