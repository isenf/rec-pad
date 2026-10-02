import pandas as pd
import re 

def filter_columns(
    df: pd.DataFrame,
    regex: str,
    class_col: str="CLASS",
) -> pd.DataFrame:
    """
    
    """
    cols = [c for c in df.columns if re.search(regex, c)]
    return df[list(set(cols + [class_col]))]


def get_non_redundant(
    df: pd.DataFrame,
    threshold: float=0.8,
    class_col: str="CLASS"
) -> list[str]:
    """
    
    """
    cols = sorted(col for col in df.columns if col!=class_col)
    corr = df[cols].corr().abs()
    keep = []

    for col in cols:
        if not any(corr.loc[col, k] > threshold for k in keep):
            keep.append(col)

    return sorted(keep)


def vote_features(
    data: dict[str, pd.DataFrame],
    threshold: int=0.8,
    class_col: str="CLASS",
    min_votes:int=None
) -> tuple:
    """
    
    """
    per_specie = {s: get_non_redundant(df, threshold, class_col)
                  for s, df in data.items()}

    all_feats = sorted({
        c 
        for df in data.values()
        for c in df.columns if c!= class_col
    })

    all_feats = sorted({f for kept in per_specie.values() for f in kept})
    votes = pd.DataFrame(
        {s: [f in per_specie[s] for f in all_feats] for s in per_specie},
        index=all_feats
    )

    votes["n_votes"] = votes.sum(axis=1)

    k = min_votes if min_votes is not None else len(data)-1
    kept = sorted(votes.index[votes["n_votes"] >= k])

    return kept, votes


def select_features(
    data: dict[str, pd.DataFrame],
    threshold: float=0.8,
    class_col: str="CLASS",
    min_votes: int=None,
    return_votes: bool=False,
) -> dict[str, pd.DataFrame] | tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    
    """
    kept, votes = vote_features(data, threshold, class_col, min_votes)
    data_filtered = {s: df[kept+[class_col]].copy() for s, df in data.items()}
    
    return (data_filtered, votes) if return_votes else data_filtered


def minmax_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    
    """
    return (df - df.min())/(df.max()-df.min())
