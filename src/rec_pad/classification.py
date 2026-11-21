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
    out_dir: Path
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
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
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
        ("kNN", knn_pipe, knn_grid),
        ("GradientBoosting", gb_pipe, gb_grid)
    ]


def _encode_y(
    y: pd.Series|np.ndarray
) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


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
