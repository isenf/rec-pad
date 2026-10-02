import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os


def _as_figure(
    obj: Figure|Axes
) -> Figure:
    if isinstance(obj, Figure):
        return obj
    if isinstance(obj, Axes):
        return obj.figure
    raise TypeError("expected matplot's Figure or Axes, got", type(obj).__name__)


def load_species(
    path: str
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    files = sorted(os.listdir(path=path))
    species = [file.split(".")[0] for file in files]
    data = {}
    
    for specie, file in zip(species, files):
        data[specie] = pd.read_parquet(f"{path}/{file}")

    return data, species


def _make_dir(path: str) -> None:
    Path(path).mkdir(mode=0o777, 
                     parents=True, 
                     exist_ok=True)
    return path


def save_fig(
    fig: Figure,
    path: str,
    file_name: str,
    dpi: int=400
) -> None:
    fig = _as_figure(fig)
    path = _make_dir(path=path)
    fig.savefig(fname=f"{path}/{file_name}", dpi=dpi)

