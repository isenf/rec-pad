import pandas as pd
import numpy as np
from rec_pad.utils.proba import prior_proba

def make_bins(
    df: pd.DataFrame,
    feature: str,
    n_bins: int=20
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """
    Partitions the values into n_bins.
    Δ_b

    Parameters
    ----------
    df: pd.DataFrame
        Input data.
    feature: str
        Feature name.
    n_bins: int, optional
        Number of bins. The default is 20.
    
    Returns
    -------
    tuple[pd.Series, np.ndarray, np.ndarray]
        Bins, edges and centers.
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
    Bin per class (raw values, not normalized).
    x∈Δ_b & class=w_i

    Parameters
    ----------
    df: pd.DataFrame
        Input data.
    binned: pd.Series
        Bins.
    class_col: str
        The class column name.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with bins per class.
    """
    return pd.crosstab(binned, df[class_col])


def priors(
    df: pd.DataFrame,
    class_col: str,
) -> pd.Series:
    """
    Compute prior probability (returns a pandas series).
    P(w_i) -> all classes sums to 1.

    Parameters
    ----------
    pd: pd.DataFrame
        Input data.
    class_col:
        The class column name.

    Returns
    -------
    pd.Series
        Prior probabilities.
    """
    return pd.Series(prior_proba(df, class_col)).sort_index()


def likelihood(
    counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the likelihood.
    P(x∈Δ_b| w_i)

    Parameters
    ----------
    counts: pd.Dataframe
        Raw counts.
    
    Returns
    -------
    pd.DataFrame
        Likelihood.
    """
    return counts/counts.sum(axis=0)  # each column sum to 1 -> normalization


def joint(
    likelihood: pd.DataFrame,
    prior: pd.Series
) -> pd.DataFrame:
    """
    Compute the numerator of Bayes' Theorem (likelihood ⋅ prior).
    P(x∈ Δ_b,w_i)=P(x∈Δ_b| w_i)⋅P(w_i)

    Parameters
    ----------
    likelihood: pd.DataFrame
        Likelihood.
    prior: pd.Series
        Prior probability.

    Returns
    -------
    pd.DataFrame
        Numerator of Bayes' Theorem.
    """
    return likelihood.mul(prior, axis=1)   # whole matrix sum to 1


def evidence(
    joint: pd.DataFrame
) -> pd.Series:
    """
    Compute the evidence (i.e. normalization factor).
    P(x)

    Parameters
    ----------
    joint: pd.Dataframe
        Numerator of Bayes' Theorem.

    Returns
    -------
    pd.Series
        Evidence.
    """
    return joint.sum(axis=1)


def posterior(
    joint: pd.DataFrame,
    evidence: pd.Series,
) -> pd.DataFrame:
    """
    Compute the posterior probability.

    Parameters
    ----------
    joint: pd.DataFrame
        Numerator of Bayes' Theorem.
    evidence: pd.Series
        Evidence.
    
    Returns
    -------
    pd.DataFrame
        Posterior probability.
    """
    return joint.div(evidence,axis=0)


def pdf_posterior(
    df: pd.DataFrame,
    feature: str,
    class_col: str,
    n_bins: int=20,
) -> dict:
    """
    Uses all functions above to compute the posterior and pdf.

    df: pd.DataFrame
        Input data.
    feature: str
        Feature column name.
    class_col: str
        Class column name.
    n_bins: int, optional
        Number of bins. The default is 20.

    Returns
    -------
    dict
        Hashmap containing: edges, centers, counts, prior, 
        likelihood, joint, evidence and posterior.
    """
    binned, edges, centers = make_bins(df, feature, n_bins=n_bins)
    counts_ = raw_counts(df, binned, class_col,)
    prior = priors(df, class_col)
    likelihood_ = likelihood(counts_)
    joint_ = joint(likelihood_, prior)
    evidence_ = evidence(joint_)
    posterior_ = posterior(joint_, evidence_)

    return dict(edges=edges, centers=centers, counts=counts_,
                prior=prior, likelihood=likelihood_, joint=joint_,
                evidence=evidence_, posterior=posterior_)

