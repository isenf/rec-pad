import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
import os


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
    path = _make_dir(path=path)
    fig.savefig(fname=f"{path}/{file_name}", dpi=dpi)

