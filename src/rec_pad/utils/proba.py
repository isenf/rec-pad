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
    counts = df[col].value_counts()
    counts_sum = counts.sum()

    return {class_value: count/counts_sum 
            for class_value, count in counts.items()}


def event_proba(
    df: pd.DataFrame,
    event: str
) -> dict[bool: float]:
    """
    Compute the probability of a given event.

    Parameters
    ----------
    df: pd.DataFrame
        Input data.
    event: str
        Boolean event.

    Returns
    -------
    dict[bool: float]
        Event probability.
    """
    if not isinstance(event, str):
        raise ValueError(f"event must be a string")

    mask = df.eval(event)
    proba_event = mask.mean()

    return {True: proba_event,
            False: 1 - proba_event}


def intersection_proba(
    df: pd.DataFrame,
    cond1: str,
    cond2: str
) -> dict[bool: float]:
    """
    Compute the joint probability  of two given boolean events.

    Parameters
    ----------
    df: pd.DataFrame
        Input data.
    cond1: str
        First boolean condition.
    cond2: str
        Second boolean condition.
    
    Returns
    -------
    dict[bool, float]
        Joint probability.
    """
    return event_proba(df, f"({cond1}) and ({cond2})")


def union_proba(
    df: pd.DataFrame,
    cond1: str,
    cond2: str,
) -> dict[bool, float]:
    """
    Compute the union probability of two given boolean events.

    Parameters
    ----------
    df: pd.DataFrame
        Input data.
    cond1: str
        First boolean condition.
    cond2: str
        Second boolean condition.
    """
    return event_proba(df, f"({cond1}) or ({cond2})")

