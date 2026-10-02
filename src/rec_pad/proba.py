import pandas as pd
import numpy as np

def prior_proba(
    df: pd.DataFrame, 
    col: str, 
) -> dict:
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

    return (counts/counts.sum()).to_dict()


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
    p = mask.mean()

    return {True: p,
            False: 1 - p}


def intersection_proba(
    df: pd.DataFrame,
    cond1: str,
    cond2: str
) -> dict[bool: float]:
    """
    Compute the joint probability  of two given boolean events.
    P(cond1 ∧ cond2).

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
    P(cond1 v cond2).

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
        Union probability.
    """
    return event_proba(df, f"({cond1}) or ({cond2})")


def conditional_proba(
    df: pd.DataFrame,
    cond1: str,
    cond2: str
) -> dict[bool: float]:
    """
    Compute the conditional probability.
    P(cond1|cond2).

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
        Conditional probability.
    """
    p_b = event_proba(df, cond2)

    if p_b[True] == 0.0:
        raise ValueError(f"condition 2 ({cond2}) is false for every row. "
                         "p(cond1 | cond2) is undefined")

    p_joint = intersection_proba(df, cond1, cond2)
    p = float(p_joint[True]/p_b[True])

    return {True: p, False: 1.0 - p}


