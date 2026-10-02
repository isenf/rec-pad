import pandas as pd
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from rec_pad import io


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