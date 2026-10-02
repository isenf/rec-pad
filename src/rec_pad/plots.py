import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Literal
from pandas.plotting import parallel_coordinates

from rec_pad.features import minmax_dataframe


_DEFAULT_YLABELS = {
    "counts": "count",
    "prior": "P(wi)",
    "likelihood": "p(x ∈ Δb | wi)",
    "posterior": "P(wi | x ∈ Δb)",
    "joint": "p(x ∈ Δb, wi)",
    "evidence": "p(x ∈ Δb)",
    "pdf": "p(x | wi)"
}

_DEFAULT_TITLES = {
    "prior": "Prior Probability",
    "counts": "Counts per Class",
    "likelihood": "Likelihood",
    "posterior": "Posterior Probability",
    "joint": "Likelihood ⋅ Prior",
    "evidence": "Evidence",
    "pdf": "Class-Conditional PDF"
}

_HIST_KEYS = ("prior", "counts")
_CURVE_KEYS = ("likelihood", "posterior", "joint", "evidence", "pdf")
_DEFAULT_KEYS = ("prior", "counts", "likelihood", "joint", 
                 "evidence", "posterior")


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

    order = sorted(df[class_col].unique())

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
        order=order,
        hue_order=order,
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
        squeeze=False,
        constrained_layout=True
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

    fig.suptitle("Violin plot grid",
                 fontsize=15,
                 fontweight="bold")
    # fig.tight_layout()
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
        corner=False,
        hue_order=sorted(df[class_col].unique())
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
        df_copy = minmax_dataframe(df_copy.drop(columns=class_col))
        df_copy[class_col] = df[class_col]

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


def _format_decimals(
    interval: pd.Interval,
    decimals: int=2,
) -> str:
    try:
        left = round(float(interval.left), decimals)
        right = round(float(interval.right), decimals)
    except AttributeError:
        return str(interval)

    return f"({left}, {right}]"


def _find_dec_boundary(
    data: pd.DataFrame,
    x: np.ndarray,
    n_dense: int = 500
) -> list[float]:
    x = np.asarray(x)
    class_names = data.columns.to_numpy()
    bounds = []

    x_dense = np.linspace(x.min(), x.max(), n_dense)
    y_dense = np.vstack([
        np.interp(x_dense, x, data[c].to_numpy(dtype=float))
                  for c in class_names
    ])
    winners = class_names[np.argmax(y_dense, axis=0)]

    for i in range(1, len(x_dense)):
        if winners[i] != winners[i-1]:  # dominant class switches
            c_prev = winners[i-1]
            c_cur = winners[i]

            idx_prev = int(np.where(class_names == c_prev)[0][0])
            idx_cur = int(np.where(class_names == c_cur)[0][0])

            y_prev_a = y_dense[idx_prev, i-1]
            y_prev_b = y_dense[idx_cur, i-1]
            y_cur_a = y_dense[idx_prev, i]
            y_cur_b = y_dense[idx_cur, i]

            da = y_cur_a-y_prev_a
            db = y_cur_b-y_prev_b
            n = y_prev_b-y_prev_a
            d = da-db

            t = 0.5 if d == 0 else np.clip(n/d, 0.0, 1.0)
            xb = x_dense[i-1] + t*(x_dense[i]-x_dense[i-1])
            bounds.append(xb)

    return bounds


def plot_binned_curves(
    result: dict,
    key: Literal["likelihood", "posterior", "joint", "counts", "evidence", "pdf"],
    feature: str,
    title: str|None=None,
    marker: str|None=None,
    y_label:str | None=None,
    y_lim: tuple[float, float]|None=None,
    ax: Axes = None,
    show_decision: bool=True,
    step: bool=False,
) -> Axes:
    """
    
    """
    if key not in _CURVE_KEYS:
        raise ValueError(f"unknown key value: {key}, must be in {_CURVE_KEYS}")
    
    ax = ax or plt.gca()
    data = result[key]
    x = np.array([interval.mid for interval in data.index])
    lines = []

    if key == "evidence":
        ax.plot(x, data)
    else:
        for c in data.columns:
            y = data[c].to_numpy()
            if step:
                (l,) = ax.step(x, y, where="mid", label=str(c))
            else:
                (l,) = ax.plot(x, y, 
                               label=str(c), marker=marker)
            lines.append(l)

    if show_decision and key in ("posterior", "joint"):
        bounds = _find_dec_boundary(data, x)
        for i, xb in enumerate(bounds):
            ax.axvline(
                xb,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="decision boundary" if i == 0 else None
            )

    ax.set_xlabel(xlabel=feature)
    ax.set_ylabel(ylabel=y_label or _DEFAULT_YLABELS[key])
    if title is not None:
        ax.set_title(title)
    if y_lim is not None and isinstance(y_lim, tuple[float, float]):
        ax.set_ylim(*y_lim)
    if lines:
        ax.legend()
    return ax


def plot_binned_hist(
    result: dict,
    key: Literal["prior", "counts"],
    title: str|None=None,
    ylabel: str = None,
    xlabel: str=None,
    bar_width: float=1.0,
    interval_decimals: int=2,
    ax: Axes = None,
):
    if key not in _HIST_KEYS:
        raise ValueError(f"unknown key value: {key}, must be in {_HIST_KEYS}")
    
    ax = ax or plt.gca()
    data = result[key]

    data.plot(kind="bar", 
              ax=ax, 
              width=bar_width,
              xlabel=xlabel,
              ylabel=ylabel or _DEFAULT_YLABELS[key])
    

    if key == "prior":
        ax.set(ylim=(0, 1.1))
        ax.bar_label(ax.containers[0], fmt="%.2f", rotation=0)
        ax.set_xticklabels(data.index, rotation=0)
    else:
        new_labels = [_format_decimals(idx, decimals=interval_decimals) 
                      for idx in data.index]
        ax.set_xticklabels(new_labels,
                           rotation=45)
        ax.legend(title="class")

    if title is not None:
        ax.set_title(title)

    return ax


def plot_pdf(
    result: dict,
    feature: str,
    title: str|None=None,
    step: bool=False,
    marker: str|None=None,
    y_label: str|None=None,
    figsize: tuple[float, float]=(8, 6)
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)

    plot_binned_curves(result=result, 
                       key="pdf",
                       feature=feature,
                       marker=marker,
                       step=step,
                       y_label=y_label,
                       ax=ax
                       )
    ax.set_title(title if title is not None else _DEFAULT_TITLES["pdf"])

    return fig


def plot_posterior(
    result: dict,
    feature: str,
    keys: list[str]=_DEFAULT_KEYS,
    n_cols: int=3,
    figsize: tuple[float, float]=None
) -> Figure:
    n_rows = (len(keys)+n_cols-1)//n_cols
    if figsize is None:
        figsize = (9.0 * n_cols, 7 * n_rows)

    fig, axes = plt.subplots(nrows=n_rows, 
                             ncols=n_cols, 
                             figsize=figsize,
                             constrained_layout=True)
    axes = axes.ravel()

    for ax, key in zip(axes, keys):
        if key in _HIST_KEYS:
            plot_binned_hist(result, 
                             key=key, 
                             ax=ax,
                             title=_DEFAULT_TITLES[key],
                             xlabel=feature,
                             bar_width=0.85 if key=="prior" else 1.0,
                             )
        else:
            plot_binned_curves(result,
                               feature=feature,
                               key=key,
                               title=_DEFAULT_TITLES[key],
                               ax=ax)

    fig.suptitle(f"Bayesian Decision Theory - {feature}",
                 fontsize=16,
                 fontweight="bold")

    return fig
