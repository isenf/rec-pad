import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
from .features import minmax_dataframe
from pandas.plotting import parallel_coordinates


def corr_heatmap(
    df: pd.DataFrame,
    class_col: str="CLASS",
    ax: Axes=None,
    figsize: tuple[float, float]=(20, 16),
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


def violin_plot(
    df: pd.DataFrame,
    feature: str,
    class_col: str="CLASS",
    ax: Axes|None=None,
    figsize:tuple[float,float]=(6, 5),
    title: str=None,
    ylim: tuple[float, float]|None=None,
    gap: float=0.0,
    width: float=1.0,
) -> Axes:
    if class_col not in df.columns:
        raise KeyError(f"the class_col '{class_col}' must be in the dataframe")
    if feature not in df.columns:
        raise KeyError(f"the feature '{feature}' must be in the dataframe")

    n_classes = len(df[class_col].unique())
    use_split = n_classes==2

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=df,
        x=class_col,
        y=feature,
        hue=class_col if use_split else None,
        split=use_split,
        inner="quart",
        gap=gap,
        cut=2,
        width=width,
        ax=ax,
        legend=True
    )

    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)

    return ax


def violin_grid(
    df: pd.DataFrame,
    features: list[str],
    class_col: str="CLASS",
    ncols: int=3,
    figsize_cell: tuple[float, float]=(4, 3.5),
    sharey: bool=False
) -> Figure:
    n=len(features)
    ncols=min(ncols, n)
    nrows = int(np.ceil(n/ncols))

    df_copy = df.copy()
    if sharey:
        df_copy = minmax_dataframe(df=df_copy.drop(columns=class_col))
        df_copy[class_col] = df[class_col]

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_cell[0]*ncols, figsize_cell[1]*nrows),
        sharey=sharey,
        squeeze=False
    )

    for ax, feature in zip(axes.flat, features):
        violin_plot(
            df_copy, 
            feature, 
            class_col=class_col,
            ax=ax,
            title=feature,
            width=1.0,
            gap=0.1)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def pair_plot(
    df: pd.DataFrame,
    class_col: str="CLASS",
    features: list[str]|None=None,
    max_features: int=12,
    title: str|None=None
) -> Figure:
    if class_col not in df.columns:
        raise KeyError(f"the class_col '{class_col}' must be in the dataframe")

    feats_cols = features or [c for c in df.columns if c!= class_col]
    feats_cols = feats_cols[:max_features]
    sub_df = df[feats_cols + [class_col]]

    ax = sns.pairplot(
        sub_df,
        hue=class_col,
        diag_kind="kde",
        corner=False
    )

    if title:
        ax.figure.suptitle(title)

    return ax.figure


def parallel_coords(
    df: pd.DataFrame,
    class_col: str="CLASS",
    figsize:tuple[float,float]=(6, 5),
    colormap: str|None=None,
    color: str|None=None,
    alpha:float=0.5,
    lw: float=0.5,
    ax: Axes|None=None,
    title: str|None=None,
    normalize:bool=True,
) -> Axes:
    if class_col not in df.columns:
        raise KeyError(f"the class_col '{class_col}' must be in the dataframe")

    feat_cols = [c for c in df.columns if c!= class_col]
    df_copy = df.copy()
    if normalize:
        df_copy = minmax_dataframe(df_copy)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    kw = {"ax": ax, "alpha": alpha, "lw": lw}
    if color is not None:
        kw["color"] = tuple(color)
    elif colormap is not None:
        kw["colormap"] = colormap

    parallel_coordinates(
        df_copy,
        class_col,
        **kw)

    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=45, ha="right")
    ax.set_ylabel("normalized value" if normalize else "value")
    
    if title:
        ax.set_title(title)

    ax.legend()
    return ax
