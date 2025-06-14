# Penulis: Tim Scikit-learn + Penyesuaian oleh [Nama Kamu]
# Tujuan: Membandingkan berbagai algoritma clustering unsupervised learning
# Data: Data sintetis seperti blobs, moons, circles, dll.

import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# =======================
# 1. Generate synthetic datasets
# =======================
n_samples = 500
seed = 30
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# Dataset aniso = transformasi linier untuk membuat cluster tidak simetris
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# Dataset dengan variasi standar deviasi antar cluster
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# =======================
# 2. Set default parameter untuk tiap algoritma
# =======================
default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "allow_single_cluster": True,
    "hdbscan_min_cluster_size": 15,
    "hdbscan_min_samples": 3,
    "random_state": 42,
}

datasets = [
    (noisy_circles, {"damping": 0.77, "preference": -240, "quantile": 0.2, "n_clusters": 2}),
    (noisy_moons, {"damping": 0.75, "preference": -220, "n_clusters": 2}),
    (varied, {"eps": 0.18, "n_neighbors": 2}),
    (aniso, {"eps": 0.15, "n_neighbors": 2}),
    (blobs, {}),
    (no_structure, {}),
]

# Untuk menyimpan hasil evaluasi
evaluation_results = []

# =======================
# 3. Iterasi setiap dataset dan setiap algoritma
# =======================
plt.figure(figsize=(20, 14))
plot_num = 1

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    params = default_base.copy()
    params.update(algo_params)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    # Estimasi bandwidth untuk MeanShift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # Konektivitas untuk clustering hierarki (Ward, Agglomerative)
    connectivity = kneighbors_graph(X, n_neighbors=params["n_neighbors"], include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)

    # Daftar algoritma clustering
    clustering_algorithms = [
        ("MiniBatchKMeans", cluster.MiniBatchKMeans(n_clusters=params["n_clusters"], random_state=params["random_state"])),
        ("AffinityPropagation", cluster.AffinityPropagation(damping=params["damping"], preference=params["preference"], random_state=params["random_state"])),
        ("MeanShift", cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)),
        ("SpectralClustering", cluster.SpectralClustering(n_clusters=params["n_clusters"], eigen_solver="arpack", affinity="nearest_neighbors", random_state=params["random_state"])),
        ("Ward", cluster.AgglomerativeClustering(n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity)),
        ("Agglomerative", cluster.AgglomerativeClustering(linkage="average", metric="cityblock", n_clusters=params["n_clusters"], connectivity=connectivity)),
        ("DBSCAN", cluster.DBSCAN(eps=params["eps"])),
        ("OPTICS", cluster.OPTICS(min_samples=params["min_samples"], xi=params["xi"], min_cluster_size=params["min_cluster_size"])),
        ("BIRCH", cluster.Birch(n_clusters=params["n_clusters"])),
        ("GaussianMixture", mixture.GaussianMixture(n_components=params["n_clusters"], covariance_type="full", random_state=params["random_state"])),
    ]

    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            algorithm.fit(X)
        t1 = time.time()

        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        # Hitung Silhouette Score jika memungkinkan
        if len(set(y_pred)) > 1 and len(set(y_pred)) < len(X):
            score = silhouette_score(X, y_pred)
        else:
            score = -1  # Tidak bisa dihitung jika cluster cuma 1 atau terlalu banyak

        # Simpan hasil evaluasi untuk laporan
        evaluation_results.append({
            "dataset": i_dataset,
            "algorithm": name,
            "clusters": len(set(y_pred)),
            "silhouette": score,
            "time": t1 - t0
        })

        # Plot hasil clustering
        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, fontsize=12)

        colors = np.array(list(islice(cycle(["#377eb8", "#ff7f00", "#4daf4a",
                                             "#f781bf", "#a65628", "#984ea3",
                                             "#999999", "#e41a1c", "#dede00"]),
                                      int(max(y_pred) + 1))))
        colors = np.append(colors, ["#000000"])  # Warna hitam untuk outlier

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
        plt.xticks([])
        plt.yticks([])
        plt.text(0.99, 0.01, f"{t1 - t0:.2f}s\nS: {score:.2f}", transform=plt.gca().transAxes,
                 size=9, horizontalalignment="right")
        plot_num += 1

plt.suptitle("Perbandingan Berbagai Algoritma Clustering\n(Skikit-Learn Examples + Penyesuaian untuk Laporan)", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Menampilkan 5 hasil evaluasi pertama
import pandas as pd
pd.DataFrame(evaluation_results).head()
