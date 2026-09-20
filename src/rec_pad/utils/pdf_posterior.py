import pandas as pd
import numpy as np
from rec_pad.utils.proba import prior_proba

def make_bins(
    df: pd.DataFrame,
    feature: str,
    n_bins: int=20
) -> tuple[pd.Series,]:
    """
    Δ_b
    """
    binned = pd.cut(df[feature], bins=n_bins, include_lowest=True)  # categorical series
    intervals = binned.cat.categories
    edges = np.array([interval.left for interval in intervals]+
                     [intervals[-1].right])
    centers = (edges[:-1]+edges[1:]) /2

    return binned, edges, centers


def raw_counts(
    df: pd.DataFrame,
    binned: pd.Series,
    class_col: str,
) -> pd.DataFrame:
    """
    bin per class
    x∈Δ_b & class=w_i
    """
    return pd.crosstab(binned, df[class_col])


def priors(
    df: pd.DataFrame,
    class_col: str,
) -> pd.Series:
    """
    prior proba but with series
    P(w_i) -> all classes sums to 1
    """
    return pd.Series(prior_proba(df, class_col)).sort_index()


def likelihood(
    counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    P(x∈Δ_b| w_i)
    """
    return counts/counts.sum(axis=0)  # each column sum to 1 -> normalization


def joint(
    likelihood: pd.DataFrame,
    priori: pd.Series
) -> pd.DataFrame:
    """
    P(x∈ Δ_b,w_i)=P(x∈Δ_b| w_i)⋅P(w_i)
    """
    return likelihood.mul(priori, axis=1)   # whole matrix sum to 1


def evidence(
    joint: pd.DataFrame
) -> pd.Series:
    """
    P(x)
    """
    return joint.sum(axis=1)


def posterior(
    joint: pd.DataFrame,
    evidence: pd.Series,
) -> pd.DataFrame:
    return joint.div(evidence,axis=0)


def pdf_posteerior(
    df: pd.DataFrame,
    feature: str,
    class_col: str,
    n_bins: int=20,
) -> dict:
    binned, edges, centers = make_bins(df, feature, n_bins=n_bins)
    counts_ = raw_counts(df, binned, class_col,)
    prior = priors(df, class_col)
    likelihood_ = likelihood(counts_)
    joint_ = joint(likelihood_, prior)
    evidence_ = evidence(joint_)
    posterior_ = posterior(joint_, evidence_)

    return dict(edges=edges, centers=centers, counts=counts_,
                priors=prior, likelihood=likelihood_, joint=joint_,
                evidence=evidence_, posterior=posterior_)

