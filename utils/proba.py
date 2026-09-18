import pandas as pd
import numpy as np

def prior_proba(
    df: pd.DataFrame, 
    col: str, 
) -> dict | None:
    """
    Compute the priori probability.

    Parameters
    ----------
    df: pd.DataFrame
        Input data.
    col: str
        Column name.

    Returns
    -------
    dict
        Mapping for {class_value: probability}.

    """
    if col not in df.columns:
        raise KeyError(f"{col} not found in the dataframe")
    counts = dict(df[col].value_counts())
    counts_sum = np.sum(counts.values())

    return {class_value: count/counts_sum 
            for class_value, count in counts.items()}





