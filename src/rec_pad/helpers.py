import pandas as pd
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from rec_pad import io, features, clustering, plots


def print_proba(
    specie: str,
    proba_dict: dict
) -> None:
    """
    Prints the probability result.

    Parameters
    ----------
    specie: str
        Specie name.
    proba_dict: dict
        The probability return dict.
    """
    print(f"\n{specie}")
    for cls, p in proba_dict.items():
        print(f"{cls}: {p:.4f}")


def print_proba_by_species(
    data: dict[str, pd.DataFrame],
    species: list[str],
    title: str,
    func: Callable,
    **kwargs
) -> None:
    """
    Calculates and prints the probability by species.

    Parameters
    ----------
    data: dict[str, pd.DataFrame]
        The data dictionary.
    species: list[str]
        Species names list.
    title: str
        Title.
    func: Callable
        Function.
    **kwargs
        Function arguments.
    """
    print(f"{title}")
    for specie in species:
        p = func(df=data[specie], **kwargs)
        print_proba(f"{specie}", p)


def save_and_close(
    fig: Figure,
    subdir: str,
    file_name: str,
    path_output: str="../../output"
) -> None:
    """
    Save the figure and close the figure.

    Parameters
    ----------
    fig: Figure
        Figure.
    subdir: str
        Subdir path from a path_output.
    file_name: str
        The file name.
    path_output: str
        Path output.
    """
    io.save_fig(fig=fig, path=f"{path_output}/{subdir}", 
                file_name=file_name, dpi=250)
    plt.close(fig)


def run_hiearchical(
    data_dicts: dict,
    metric: str,
    class_col: str="CLASS",
    n_clusters: int=2,
    path_output: str=None
) -> dict:
    results = {
        "scaled": {},
        "minmax": {}
    }

    for scale_type, data_dict in data_dicts.items():
        for specie in data_dict.keys():
            X, y = features.prepare_xy(data_dict[specie], class_col=class_col)
            y_enc = features.encode_labels(y)

            labels, _ = clustering.hierarchical_clustering(
                X=X, metric=metric, method="average", n_clusters=n_clusters
            )
            results[scale_type][specie] = clustering.evaluate_clustering(
                X=X, labels=labels, y_true=y_enc
            )
            if path_output is not None:
                 fig = plots.plot_contingency_matrix(y_true=y_enc, 
                                                     labels=labels, 
                                                     true_labels_names=np.unique(y),
                                                     title=f"Contingency Matrix - {metric} {scale_type}",
                                                     normalize_by="col"
                                                     )
                 save_and_close(fig=fig, path_output=path_output, 
                                subdir=f"clustering/hierarchical/normalized_col/{metric}/{scale_type}",
                                file_name=f"{specie}.jpg")

    return results


def run_kmeans(
    data_dicts: dict,
    class_col: str="CLASS",
    random_state: int=23,
    n_clusters: int=2,
    path_output: str=None
) -> dict:
    results = {
        "scaled": {},
        "minmax": {}
    }

    for scale_type, data_dict in data_dicts.items():
        for specie in data_dict.keys():
            X, y = features.prepare_xy(data_dict[specie], class_col=class_col)
            y_enc = features.encode_labels(y)

            labels, _ = clustering.kmeans_clustering(
                X=X, random_state=random_state, n_init=10, init="k-means++", n_clusters=n_clusters
            )
            results[scale_type][specie] = clustering.evaluate_clustering(
                X=X, labels=labels, y_true=y_enc
            )
            if path_output is not None:
                 fig = plots.plot_contingency_matrix(y_true=y_enc, 
                                                     labels=labels, 
                                                     true_labels_names=np.unique(y),
                                                     title=f"Contingency Matrix - {scale_type}",
                                                     normalize_by="col"
                                                     )
                 save_and_close(fig=fig, path_output=path_output, 
                                subdir=f"clustering/kmeans/contingency_matrix/normalized_col/{scale_type}",
                                file_name=f"{specie}.jpg")

    return results


def metrics_to_dataframe(
    results_dict: dict[str, dict]
) -> pd.DataFrame:
    records = []

    for scale_type, specie_dict in results_dict.items():
        for specie, metrics_dict in specie_dict.items():
            row = {"scale": scale_type, "specie": specie}
            row.update(metrics_dict)
            records.append(row)
    df = pd.DataFrame(records)
    return df.set_index(["scale", "specie"])
