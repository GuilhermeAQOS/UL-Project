from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class APCluster:
    indices: list[int]
    centroid_raw: FloatArray
    centroid_std: FloatArray
    size: int
    scatter_pct: float


def compute_feature_statistics(
    X: FloatArray,
    use_unit_ranges: bool = False,
) -> tuple[FloatArray, FloatArray, float]:
    X = np.asarray(X, dtype=np.float64)
    g_mean = X.mean(axis=0)
    if use_unit_ranges:
        feature_scale = np.ones(X.shape[1], dtype=np.float64)
    else:
        feature_scale = X.max(axis=0) - X.min(axis=0)
        feature_scale = np.where(feature_scale == 0, 1.0, feature_scale)
    Y = (X - g_mean) / feature_scale
    D = float(np.sum(Y ** 2))
    return g_mean, feature_scale, D


def normalized_squared_distances(
    X: FloatArray,
    indices: list[int],
    scales: FloatArray,
    reference: FloatArray,
) -> FloatArray:
    if len(indices) == 0:
        return np.array([], dtype=np.float64)
    return np.sum(((X[indices] - reference) / scales) ** 2, axis=1)


def cluster_centroid(
    X: FloatArray,
    indices: list[int],
) -> FloatArray:
    if len(indices) == 0:
        raise ValueError("Cannot compute centroid of empty cluster.")
    return X[indices].mean(axis=0)


def separate_cluster(
    X: FloatArray,
    indices: list[int],
    scales: FloatArray,
    a: FloatArray,
    b: FloatArray,
) -> list[int]:
    if len(indices) == 0:
        return []
    d_a = normalized_squared_distances(X, indices, scales, a)
    d_b = normalized_squared_distances(X, indices, scales, b)
    mask = d_a < d_b
    selected = np.asarray(indices)[mask]
    return sorted(selected.tolist())


def extract_anomalous_cluster(
    X: FloatArray,
    indices: list[int],
    scales: FloatArray,
    mean: FloatArray,
    initial_centroid: FloatArray,
    seed_index: int,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> tuple[list[int], FloatArray]:
    c = np.asarray(initial_centroid, dtype=np.float64).copy()
    S_prev: list[int] = []
    for _ in range(max_iter):
        S = separate_cluster(X, indices, scales, c, mean)
        if len(S) == 0:
            S = [seed_index]
        c_new = cluster_centroid(X, S)
        if S == S_prev:
            return S, c_new
        if np.linalg.norm(c_new - c) <= tol:
            return S, c_new
        c = c_new
        S_prev = S
    return S, c


def ikmeans_initialize(
    X: FloatArray,
    min_cluster_size: int,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    use_unit_ranges: bool = False,
) -> tuple[list[APCluster], FloatArray]:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    g_mean, feature_scale, D = compute_feature_statistics(X, use_unit_ranges=use_unit_ranges)
    remains: list[int] = list(range(n))
    ap_clusters: list[APCluster] = []
    while len(remains) > 0:
        d_to_mean = normalized_squared_distances(X, remains, feature_scale, g_mean)
        farthest_pos = int(np.argmax(d_to_mean))
        seed_index = remains[farthest_pos]
        seed = X[seed_index]
        S, c_raw = extract_anomalous_cluster(
            X=X,
            indices=remains,
            scales=feature_scale,
            mean=g_mean,
            initial_centroid=seed,
            seed_index=seed_index,
            tol=tol,
            max_iter=max_iter,
        )
        c_std = (c_raw - g_mean) / feature_scale
        if D > 0:
            scatter_pct = 100.0 * len(S) * float(np.sum(c_std ** 2)) / D
        else:
            scatter_pct = 0.0
        ap_clusters.append(APCluster(
            indices=sorted(S),
            centroid_raw=c_raw,
            centroid_std=c_std,
            size=len(S),
            scatter_pct=scatter_pct,
        ))
        remains = sorted(set(remains) - set(S))
    retained = [cl for cl in ap_clusters if cl.size >= min_cluster_size]
    if len(retained) == 0:
        raise ValueError(
            f"No anomalous cluster satisfies the minimum size threshold "
            f"min_cluster_size={min_cluster_size}."
        )
    init_centroids = np.vstack([cl.centroid_std for cl in retained])
    return ap_clusters, init_centroids

        
        
