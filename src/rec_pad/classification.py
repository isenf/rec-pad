import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS    
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
)

from rec_pad import features, io, helpers


def _evaluate(
    specie: str,
    classifier_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_dir: str
) -> dict:
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1_mac = f1_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0)
    f1_wgt = f1_score(y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0)
    print(f"{classifier_name} - {specie}")
    print(f"accuracy: {acc}")
    print(f"f1 macro: {f1_mac}")
    print(f"f1 weighted: {f1_wgt}")
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=labels,
                                digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_df = pd.DataFrame(
        cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]
    )
    print("confusion matrix")
    print(cm_df)
    if out_dir is not None:
        out_dir = Path(out_dir)
        io._make_dir(path=out_dir)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, cmap="Blues", values_format="d"
        )
        ax.set_title(f"{classifier_name} - {specie}")
        fig.tight_layout()
        helpers.save_and_close(fig, subdir=out_dir, file_name=f"{specie}.jpg")
    return {
        "specie": specie,
        "accuracy": acc,
        "f1_macro": f1_mac, 
        "f1_weighted": f1_wgt
    }


def _models_and_grids(
    seed:int=23
):
    dummy = DummyClassifier(strategy="prior", random_state=seed)
    dummy_grid = {}
    knn_pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", KNeighborsClassifier())
    ])
    knn_grid = {
        "model__n_neighbors": [3, 5, 7, 11, 15, 21],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2]
    }
    gb_pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", GradientBoostingClassifier(random_state=seed))
    ])
    gb_grid = {
        "model__n_estimators": [100, 200, 400],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.8, 1.0],
        "model__min_samples_leaf": [1, 5]
    }

    return [
        ("Dummy", dummy, dummy_grid),
        ("kNN", knn_pipe, knn_grid),
        ("GradientBoosting", gb_pipe, gb_grid)
    ]


def _encode_y(
    y: pd.Series|np.ndarray
) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def _prepare_species_data(
    data: dict,
    species: list[str],
    class_col: str="CLASS",
    test_size: float=0.2,
    seed:int=23
) -> dict:
    prepared = {}
    for specie in species:
        X, y = features.prepare_xy(data[specie], class_col=class_col)
        y_enc, le = _encode_y(y)
        X_train, X_test, y_train, y_test = split_data(
            X, y_enc, test_size=test_size, random_state=seed
        )
        prepared[specie] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "labels": list(le.classes_)
        }
    return prepared


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size:float=0.2,
    random_state:int=23
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def run_all_models_species(
    data: dict,
    species: list[str],
    out_dir: str,
    class_col: str="CLASS",
    scoring: str="f1_weighted",
    test_size: float=0.2,
    cv:int=10,
    seed: int = 23,
    tag: str="all"
) -> pd.DataFrame:
    prepared = _prepare_species_data(
        data=data, species=species, class_col=class_col, test_size=test_size,
        seed=seed
    )
    all_results = []

    for classifier_name, pipe, grid in _models_and_grids(seed=seed):
        for specie in species:
            d = prepared[specie]
            gs = GridSearchCV(
                estimator=pipe, param_grid=grid, scoring=scoring, cv=cv,
                n_jobs=-1, refit=True
            )
            gs.fit(d["X_train"], d["y_train"])

            fold_scores = cross_val_score(
                gs.best_estimator_, d["X_train"], d["y_train"],
                cv=cv, scoring=scoring, n_jobs=-1
            )

            y_pred = gs.predict(d["X_test"])
            res = _evaluate(
                specie, classifier_name, d["y_test"], y_pred=y_pred,
                labels=d["labels"], out_dir=f"{out_dir}/{classifier_name}/{tag}"
            )
            res["feature_set"] = tag
            res["n_features"] =d["X_train"].shape[1]
            res["cv_mean"] = fold_scores.mean()
            res["cv_std"] = fold_scores.std()
            res["best_params"] = gs.best_params_ if grid else "default"
            all_results.append(res)
    
    return pd.DataFrame(all_results)
