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

_DEFAULT_TITLES = {
    "prior": "Prior Probability",
    "counts": "Counts per Class",
    "likelihood": "Likelihood",
    "posterior": "Posterior Probability",
    "joint": "Likelihood ⋅ Prior",
    "evidence": "Evidence"
}


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
    results: dict,
    key: Literal["likelihood", "posterior", "joint", "counts", "evidence"],
    feature: str,
    marker: str|None=None,
    y_label:str | None=None,
    y_lim: tuple[float, float]|None=None,
    ax: matplotlib.axes.Axes = None,
    show_decision: bool=True,
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
    if y_lim is not None and isinstance(y_lim, tuple[float, float]):
        ax.set_ylim(*y_lim)
    return ax


def plot_binned_hist(
    result: dict,
    key: Literal["prior", "counts"],
    ylabel: str = None,
    xlabel: str=None,
    bar_width: float=1.0,
    interval_decimals: int=2,
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
        ax.set(ylim=(0, 1.1))
        ax.bar_label(ax.containers[0], fmt="%.2f", rotation=0)
        ax.set_xticklabels(data.index, rotation=0)
    else:
        new_labels = [_format_decimals(idx, decimals=interval_decimals) 
                      for idx in data.index]
        ax.set_xticklabels(new_labels,
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
                             figsize=figsize,
                             constrained_layout=True)
    axes = axes.ravel()

    for ax, key in zip(axes, keys):
        if key in ("prior", "counts"):
            plot_binned_hist(result, 
                             key=key, 
                             ax=ax,
                             xlabel=feature,
                             bar_width=0.85 if key=="prior" else 1.0,
                             )
        else:
            plot_binned_curves(result,
                               feature=feature,
                               key=key,
                               ax=ax)
        ax.set_title(_DEFAULT_TITLES[key])

    fig.suptitle(f"Bayesian Decision Theory - {feature}",
                 fontsize=15,
                 fontweight="bold")

    return fig

