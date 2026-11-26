import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import joblib
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


def save_data(
    data: pd.DataFrame,
    path: str|Path,
    file_name: str
) -> None:
    _make_dir(path=path)
    path = Path(path)
    data.to_csv(path/f"{file_name}.csv", index=False)


def load_data(
    path: Path|str,
    file_format: str
) -> pd.DataFrame:
    if file_format == "csv":
        return pd.read_csv(path)
    elif file_format == "parquet":
        return pd.read_parquet(path)
    else: 
        return None


def save_classifier(
    estimator,
    out_dir: str|Path,
    specie: str,
    classifier_name: str,
    tag: str,
    labels: list[str]|None=None,
    best_params: dict|None=None,
    metrics: dict|None=None,
) -> Path:
    out_dir = Path(out_dir)/classifier_name/tag
    _make_dir(path=out_dir)
    path = f"{out_dir}/{specie}.sav"
    joblib.dump({
        "model": estimator,
        "specie": specie,
        "classifier_name": classifier_name,
        "tag": tag,
        "labels": labels,
        "best_params": best_params,
        "metrics": metrics
    }, path, compress=3)
    return path


def load_classifier(
    path: str|Path
) -> dict:
    return joblib.load(Path(path))
