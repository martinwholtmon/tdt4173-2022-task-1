import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    def __init__(self, n_clusters=2, n_iterations=20):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self._n_clusters = n_clusters
        self._n_iterations = n_iterations
        self._cluster = []
        self._centroids = []
        self._mean = np.ndarray
        self._std = np.ndarray

    def fit(self, X):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        n_clusters = self._n_clusters
        n_iterations = self._n_iterations

        # Convert to array
        X = np.asarray(X)
        samples, _ = X.shape

        # Preprossessing:
        # Scaling/normalization
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        self._mean = X_mean
        self._std = X_std

        X = (X - X_mean) / X_std

        # Create initial random centroids
        centroids = get_initial_centroids(X, n_clusters)

        # k-means
        for _ in range(n_iterations):
            # Get cluster
            cluster = get_cluster_assignments(X, centroids, samples)

            # Calculate new centroids
            centroids = calulate_new_centroids(X, cluster, n_clusters, centroids)

        # Store results and reverse transformation
        self._centroids = centroids * X_std + X_mean
        self._cluster = cluster

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        X = np.asarray(X)
        X_mean = self._mean
        X_std = self._std

        X = (X - X_mean) / X_std
        centroids = (self._centroids - X_mean) / X_std

        return get_cluster_assignments(X, centroids, X.shape[0]).astype(int)
        # return self._cluster.astype(int)

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm

        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self._centroids


# --- Some utility functions


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the raw distortion measure
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()

    return distortion


def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points

    Note: by passing "y=0.0", it will compute the euclidean norm

    Args:
        x, y (array<...,n>): float tensors with pairs of
            n-dimensional points

    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points

    Args:
        x (array<m,d>): float tensor with pairs of
            n-dimensional points.
        y (array<n,d>): float tensor with pairs of
            n-dimensional points. Uses y=x if y is not given.

    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance

    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    # Intra distance
    a = D[np.arange(len(X)), z]
    # Smallest inter distance
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))


def get_cluster_assignments(X, centroids, samples) -> np.ndarray:
    cluster = np.empty((samples), float)
    for dp in range(samples):
        # For each datapoint, calulate the eucleadian distance to the centroids
        # and store the lowest distance
        e_dist = euclidean_distance(X[dp], centroids)

        # update cluster with centroid
        cluster[dp] = np.argmin(e_dist)
    return cluster


def calulate_new_centroids(data, cluster, n_clusters, centroid) -> np.ndarray:
    """Calulates the new centroid position given the current clusters. Will calculate the mean in the clusters and then return the new centroids.

    Args:
        data (np.ndarray): the dataset
        cluster (np.ndarray): cluster assignments for the dataset
        n_clusters (int): number of clusters
        centroid (np.ndarray): the current centroid locations

    Returns:
        np.ndarray: the new centroid locations
    """
    new_centroids = np.zeros(centroid.shape)
    for i in range(n_clusters):
        arr = data[np.where(cluster == i)]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        length = arr.shape[0]
        if length == 0:
            new_centroids[i] = np.array([0, 0])
        else:
            new_centroids[i] = np.array([sum_x / length, sum_y / length])
    return new_centroids


def get_initial_centroids(data, n_clusters) -> np.ndarray:
    """k-means++ implementation to get the best initial centroids:
    1. Pick random point in sample as initial centroid
    2. Calculate eucleadian distance to all data points
    3. Add the point furthest away from the clusters as new centroid
    4. Repeat 3-4 until you have reached n_clusters

    Args:
        data (np.ndarray): the sample data
        n_clusters (int): number of clusters (K=n)

    Returns:
        np.ndarray: the initial centroids calulated per k-means++ algo
    """
    samples, dimention = data.shape
    centroids = np.empty([n_clusters, dimention])
    initial_centroid = data[random.sample(range(samples), 1)]
    centroids[0] = initial_centroid

    # Initial centroid is assigned, so calucated the
    for i in range(1, centroids.shape[0]):
        cluster = np.empty((samples), float)
        for dp in range(samples):
            # For each datapoint, calulate the eucleadian distance
            # to the centroids (that exist 1..i)
            e_dist = euclidean_distance(data[dp], centroids[:i])

            # update cluster list with distance to closest centroid
            cluster[dp] = np.min(e_dist)

        # update centroid with the point furthes away from all centroids
        centroids[i] = data[np.argmax(cluster)]
    return centroids
