import pandas as pd


def basic_info(
    df: pd.DataFrame,
    class_col: str="CLASS"
) -> dict:
    """
    Compute basic information about a dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Input.
    class_col: str, optional
        Class column name. The default is "CLASS".

    Returns
    -------
    dict
        Dictionary with n_samples, n_features, classes, n_classes and
        feature_names.
    """
    class_names = sorted(df[class_col].unique().tolist())

    return {
        "n_samples": df.shape[0],
        "n_features": df.shape[1]-1,
        "classes": class_names,
        "n_classes": len(class_names),
        "feature_names": [c for c in df.columns 
                          if c!= class_col]
    }


def class_counts(
    df: pd.DataFrame,
    class_col:str="CLASS"
) -> pd.Series:
    """
    Counts samples per class, sorted by class label.

    Parameters
    ----------
    df. pd.DataFrame
        Input.
    class_col: str, optional
        Class column name. The default is "CLASS".

    Returns
    -------
    pd.Series
        Counts per class.
    """
    return df[class_col].value_counts().sort_index()


def summarize_datasets(
    data: dict[str, pd.DataFrame],
    class_col: str="CLASS"
) -> pd.DataFrame:
    """
    Summarize multiple datasets.

    Parameters
    ----------
    data: dict[str, pd.DataFrame]
        Hashmap with datasets.
    class_col: str, optional
        Class column name. The default is "CLASS".

    Returns
    -------
    pd.DataFrame
        Summary with one row per dataset, indexed by dataset name.
    """
    rows = []

    for specie, df in data.items():
        info = basic_info(df, class_col)
        counts = class_counts(df, class_col)

        rows.append({
            "specie": specie,
            "n_samples": info["n_samples"],
            "n_features": info["n_features"],
            "n_classes": info["n_classes"],
            "classes": ", ".join(map(str, info["classes"])),
            "class_counts": ", ".join(f"{key} = {val}"
                                      for key, val in counts.items())
        })

    return pd.DataFrame(rows).set_index("specie")


def missing_report(
    df: pd.DataFrame
) -> pd.Series:
    """
    Report columns with missing values.

    Parameters
    ----------
    df: pd.DataFrame
        Input.

    Returns
    -------
    pd.Series
        Number of missing values per column. 
        Empty series if there are none. 
    """
    return df.isna().sum().loc[lambda x:x>0]   # empty if None


def duplicate_count(
    df: pd.DataFrame
) -> int:
    """
    Count duplicate rows.

    Parameters
    ----------
    df: pd.DataFrame
        Input.
    
    Returns
    -------
    int
        Number of duplicate rows (keeping the first occurrence).
    """
    return int(df.duplicated(keep="first").sum())


def drop_duplicates(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Drop duplicate rows and reset the index.

    Parameters
    ----------
    df: pd.DataFrame
        Input.

    Returns
    -------
    pd.DataFrame
        DataFrame without duplicates.
    """
    df_clean = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df_clean


def datasets_desc(
    data: dict[str, pd.DataFrame]
) -> None:
    """
    Print descriptive statistics for each dataset.

    Parameters
    ----------
    data: dict[str, pd.DataFrame]
        Hashmap with datasets.
    """
    for specie, df in data.items():
        print(f"\n\n{specie}")
        print(df.describe())
