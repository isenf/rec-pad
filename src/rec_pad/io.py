import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os


def _as_figure(
    obj: Figure|Axes
) -> Figure:
    """
    Returns a matplotlib Figure from a Figure or Axes object.

    Parameters
    ----------
    obj: Figure|Axes
        A matplotlib Figure or Axes instance
    
    Returns
    -------
    Figure
        The Figure associated with obj.
    """
    if isinstance(obj, Figure):
        return obj
    if isinstance(obj, Axes):
        return obj.figure
    raise TypeError("expected matplot's Figure or Axes, got", type(obj).__name__)


def load_species(
    path: str
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    # Load all .parquet files from a given directory.

    Parameters
    ----------
    path: str
        Directory path with .parquet files.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], list[str]]
        Tuple with dict data and species names.
    """
    files = sorted(os.listdir(path=path))
    species = [file.split(".")[0] for file in files]
    data = {}
    
    for specie, file in zip(species, files):
        data[specie] = pd.read_parquet(f"{path}/{file}")

    return data, species


def _make_dir(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Parameters:
    path: str
        Directory path to create.
    
    Returns
    -------
    str
        The same path string.
    """
    Path(path).mkdir(mode=0o777, 
                     parents=True, 
                     exist_ok=True)
    return path


def save_fig(
    fig: Figure|Axes,
    path: str,
    file_name: str,
    dpi: int=400
) -> None:
    """
    Save a matplotlib figure to a file

    Parameters
    ----------
    fig: Figure|Axes
        Figure to save.
    path: str
        Directory path.
    file_name: str
        File name.
    dpi: int, optional
        Resolution in dots per inch. The default is 400.
    """
    fig = _as_figure(fig)
    path = _make_dir(path=path)
    fig.savefig(fname=f"{path}/{file_name}", dpi=dpi)

