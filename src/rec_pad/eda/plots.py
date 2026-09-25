import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns


def corr_heatmap(
    df: pd.DataFrame,
    class_col: str="CLASS",
    ax: Axes=None,
    figsize: tuple[float, float]=(10, 8),
    annot: bool=False,
    title: str=None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    elif figsize is not None:
        ax.figure.set_size_inches(*figsize)
    
    corr = df.drop(columns=class_col).corr().abs()
    
    sns.heatmap(
        corr, vmin=0, vmax=1, 
        cmap="Blues", ax=ax, annot=annot,
        square=True, 
        fmt=".2f" if annot else "",
    )

    
    ax.set_title(title or "Absolute correlation matrix")

    return ax


