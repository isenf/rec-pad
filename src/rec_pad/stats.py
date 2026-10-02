import pandas as pd


def dataset_desc(
    df: pd.DataFrame,
    class_col: str="CLASS"
) -> dict:
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
    return df[class_col].value_counts().sort_index()


def summarize_datasets(
    data: dict[str, pd.DataFrame],
    class_col: str="CLASS"
) -> pd.DataFrame:
    rows = []

    for specie, df in data.items():
        info = dataset_desc(df, class_col)
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
    return df.isna().sum().loc[lambda x:x>0]   # empty if None


def duplicate_count(
    df: pd.DataFrame
) -> int:
    return int(df.duplicated(keep="first").sum())


def drop_duplicates(
    df: pd.DataFrame
) -> pd.DataFrame:
    df_clean = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df_clean
