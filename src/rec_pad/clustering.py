import pandas as pd
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import (
    pairwise_distances, 
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def purity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate the purity score of a clustering.

    Parameters
    ----------
    y_true: np.ndarray
        True class labels.
    y_pred: np.ndarray
        Predicted cluster labels.

    Returns
    -------
    float
        The purity score (between 0 and 1).
    """
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0))/np.sum(cm)


def calc_distances(
    X: pd.DataFrame,
    labels: np.ndarray
) -> tuple[float, float]:
    """
    Calculate the average intra-cluster and inter-cluster distances.

    Parameters
    ----------
    X: pd.DataFrame
        Feature matrix.
    labels: np.ndarray
        Cluster labels for each sample.

    Returns 
    -------
    tuple[float, float]
        Tuple with avg_intra and avg_inter.
    """
    labels_unique = np.unique(labels)
    intra_dist = []

    for label in labels_unique:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            dist_matrix = pairwise_distances(cluster_points)
            intra_dist.append(np.mean(dist_matrix))
    avg_intra = np.mean(intra_dist) if len(intra_dist)>0 else 0.0

    centroids = np.array([X[labels == label].mean(axis=0) for label in labels_unique])
    if len(centroids) > 1:
        inter_dist_matrix = pairwise_distances(centroids)
        avg_inter = np.mean(inter_dist_matrix[np.triu_indices(len(centroids), k=1)])
    else:
        avg_inter = 0.0
    return avg_intra, avg_inter


def quality_cluster(
    X: pd.DataFrame, 
    labels: np.ndarray,
    return_distances: bool=False
) -> float | tuple[float, float, float]:
    """
    Calculate the quality ratio of a clustering.

    Parameters
    ----------
    X: pd.DataFrame
        Feature matrix.
    labels: np.ndarray
        Cluster labels for each sample.
    return_distances: bool, optional
        If True, returns the distances. The default is False.

    Returns
    -------
    float | tuple[float, float, float]
        If return_distances is False, returns just the quality score.
        If return_distances is True, returns (quality, intra, inter). 
    """
    intra, inter = calc_distances(X, labels)
    quality = inter/intra if intra > 0 else 0.0
    return (quality, intra, inter) if return_distances else quality


def evaluate_clustering(
    X: np.ndarray| pd.DataFrame,
    labels: np.ndarray,
    y_true: np.ndarray=None
) -> dict[str, float]:
    """
    Compute clustering metrics.

    Parameters
    ----------
    X: np.ndarray| pd.DataFrame
        Feature matrix.
    labels: np.ndarray
        Clustering labels assigned to each sample.
    y_true: np.ndarray, optional
        True class labels. The default is None.

    Returns
    -------
    dict[str, float]
        Clustering metrics.
    """
    results = {}
    unique_labels = np.unique(labels)
    valid_clusters = [cluster for cluster in unique_labels if cluster != -1]    # -1 -> noise

    if len(valid_clusters) > 1:
        mask = labels!=-1
        X_ = X[mask] if isinstance(X, np.ndarray) else X.loc[mask]
        labels_ = labels[mask]

        results["silhouette"] = silhouette_score(X=X_, labels=labels_)
        results["davies_bouldin"] = davies_bouldin_score(X=X_, labels=labels_)
        results["calinski_harabasz"] = calinski_harabasz_score(X=X_, labels=labels_)

        quality, intra, inter = quality_cluster(X, labels, return_distances=True)
        results["quality"] = quality
        results["intra_dist"] = intra
        results["inter_dist"] = inter
    else:
        results["silhouette"], results["quality"], results["intra_dist"], results["inter_dist"] = [np.nan]*4

    if y_true is not None:
        results["purity"] = purity_score(y_true, labels)
        results["nmi"] = normalized_mutual_info_score(labels_pred=labels, labels_true=y_true)
        results["ari"] = adjusted_rand_score(labels_pred=labels, labels_true=y_true)

    return results


def hierarchical_clustering(
    X: np.ndarray| pd.DataFrame,
    metric: str="euclidean",
    method: str="average",
    n_clusters: int=2,
    criterion: str="maxclust",
    linkage_matrix: np.ndarray=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Hierarchical Clustering using Scipy.

    Parameters
    ----------
    X: np.ndarray| pd.DataFrame
        Feature matrix.
    metric: str, optional
        Distance metric (e.g., "euclidean", "correlation").
        The default is "euclidean".
    method: str, optional
        Linkage method (e.g., "average", "complete", "single").
        The default is "average".
    n_clusters: int, optional
        Number of clusters. The default is 2.
    criterion: str, optional
        The criterion used to form flat clusters (e.g., "maxclust", "distance).
        The default is "maxclust".
    linkage_matrix: np.ndarray, optional
        Pre-computed linkage matrix.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple with labels and linkage matrix.
    """
    if linkage_matrix is None:
        linkage_matrix = linkage(X, method=method, metric=metric)

    labels = fcluster(linkage_matrix, t=n_clusters, criterion=criterion)
    return labels, linkage_matrix


def kmeans_clustering(
    X: np.ndarray|pd.DataFrame,
    n_clusters: int=2,
    random_state: int=23,
    n_init: int|str="auto",
    init:str="k-means++"
) -> tuple[np.ndarray, KMeans]:
    """
    Perform K-Means clustering using scikit-learn.

    Parameters
    ----------
    X: np.ndarray | pd.DataFrame
        Feature matrix.
    n_clusters: int, optional
        Number of clusters. The default is 2.
    random_state: int, optional
        Random seed. The default is 23.
    init: str, optional
        Method for initialization.
        The default is "k-means++"
    n_init: int|str, optional
        Number of times the k- means algorithm is run with different centroid seeds.
        The default is "auto".
        
    Returns
    -------
    tuple[np.ndarray, KMeans]
        Tuple with cluster labels and KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,
                    init=init, n_init=n_init)
    labels = kmeans.fit_predict(X=X)

    return labels, kmeans
