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
