import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes
from typing import Literal, Iterable


_DEFAULT_YLABELS = {
    "counts": "count",
    "prior": "P(wi)",
    "likelihood": "p(x ∈ Δb | wi)",
    "posterior": "P(wi | x ∈ Δb)",
    "joint": "p(x ∈ Δb, wi)",
    "evidence": "p(x ∈ Δb)"
}

_DEFAULT_KEYS = ("prior", "counts", 
                 "likelihood", "joint", 
                 "evidence", "posterior")


def plot_binned_curves(
    results: dict,
    key: Literal["likelihood", "posterior", "joint", "counts", "evidence"],
    feature: str,
    marker: str|None=None,
    y_label:str | None=None,
    y_lim: tuple[float, float]|None=None,
    ax: matplotlib.axes.Axes = None,
) -> matplotlib.axes.Axes:
    """
    
    """
    if key not in ("likelihood", "posterior", "joint", "counts", "evidence"):
        raise ValueError(f"unknown key value: {key}, must be "
                         "'likelihood', 'posterior', 'joint', 'counts' or 'evidence'")
    ax = ax or plt.gca()
    data = results[key]
    x = np.array([interval.mid for interval in data.index])

    if key == "evidence":
        ax.plot(x, data)
    else:
        for c in data.columns:
            ax.plot(x, data[c].to_numpy(), label=str(c), marker=marker)

    ax.set_xlabel(xlabel=feature)
    ax.set_ylabel(ylabel=y_label or _DEFAULT_YLABELS[key])
    if y_lim is not None and isinstance(y_lim, tuple[float, float]):
        ax.set_ylim(*y_lim)
    return ax


def plot_binned_hist(
    result: dict,
    key: Literal["prior", "counts"],
    ylabel: str = None,
    xlabel: str=None,
    bar_width: float=1.0,
    ax: matplotlib.axes.Axes = None,
):
    if key not in ("prior", "counts"):
        raise ValueError(f"unknown key value: {key}, must be 'prior' or 'count'")
    ax = ax or plt.gca()
    data = result[key]

    data.plot(kind="bar", 
              ax=ax, 
              width=bar_width,
              xlabel=xlabel,
              ylabel=ylabel or _DEFAULT_YLABELS[key])
    

    if key == "prior":
        ax.set(ylim=(0, 1))
        ax.bar_label(ax.containers[0], fmt="%.2f", rotation=0)
        ax.set_xticklabels(data.index, rotation=0)
    else:
        ax.set_xticklabels(data.index,
                           rotation=45)
        ax.legend(title="class")

    return ax


def plot_pdf_posterior(
    result: dict,
    feature: str,
    keys: list[str]=_DEFAULT_KEYS,
    n_cols: int=3,
    figsize: tuple[float, float]=None
):
    n_rows = (len(keys)+n_cols-1)//n_cols
    if figsize is None:
        figsize = (9.0 * n_cols, 7 * n_rows)

    fig, axes = plt.subplots(nrows=n_rows, 
                             ncols=n_cols, 
                             figsize=figsize)
    axes = axes.ravel()

    for ax, key in zip(axes, keys):
        if key in ("prior", "counts"):
            plot_binned_hist(result, 
                             key=key, 
                             ax=ax,
                             xlabel=feature,
                             bar_width=0.85 if key=="prior" else 1.0)
        else:
            plot_binned_curves(result,
                               feature=feature,
                               key=key,
                               ax=ax)

    return fig

