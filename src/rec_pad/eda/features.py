import pandas as pd
import re 

def filter_columns(
    df: pd.DataFrame,
    regex: str,
    class_col: str="CLASS",
) -> pd.DataFrame:
    cols = [c for c in df.columns if re.search(regex, c)]
    return df[list(set(cols + [class_col]))]


def get_non_redundant(
    df: pd.DataFrame,
    threshold: float=0.8,
    class_col: str="CLASS"
) -> list[str]:
    corr = df.drop(columns=class_col).corr().abs()
    keep = []

    for col in corr.columns:
        if not any(corr.loc[col, k] > threshold for k in keep):
            keep.append(col)

    return sorted(keep)


def minmax_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    return (df - df.min())/(df.max()-df.min())
