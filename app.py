import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Clustering Explorer | Muhammad Kamil Shah", page_icon="🧩", layout="wide")
st.title("Clustering Algorithms Explorer")
st.caption("Interactive companion to the clustering comparison notebook.")

with st.sidebar:
    dataset = st.selectbox("Dataset", ["Moons", "Blobs"])
    n = st.slider("Samples", 150, 800, 350, 50)
    noise = st.slider("Noise", 0.02, 0.30, 0.08, 0.01)
    algorithm = st.selectbox("Algorithm", ["Agglomerative", "DBSCAN", "OPTICS", "Mean Shift", "Gaussian Mixture"])

if dataset == "Moons":
    X, _ = make_moons(n_samples=n, noise=noise, random_state=42)
else:
    X, _ = make_blobs(n_samples=n, centers=4, cluster_std=0.75 + noise * 3, random_state=42)
X = StandardScaler().fit_transform(X)

if algorithm == "Agglomerative":
    k = st.sidebar.slider("Clusters", 2, 8, 3)
    model = AgglomerativeClustering(n_clusters=k)
elif algorithm == "DBSCAN":
    eps = st.sidebar.slider("eps", 0.05, 1.0, 0.25, 0.05)
    min_samples = st.sidebar.slider("min samples", 2, 20, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
elif algorithm == "OPTICS":
    min_samples = st.sidebar.slider("min samples", 2, 30, 8)
    model = OPTICS(min_samples=min_samples)
elif algorithm == "Mean Shift":
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=min(len(X), 300))
    model = MeanShift(bandwidth=max(bandwidth, 0.1))
else:
    k = st.sidebar.slider("Components", 2, 8, 3)
    model = GaussianMixture(n_components=k, random_state=42)

start = perf_counter()
labels = model.fit_predict(X) if hasattr(model, "fit_predict") else model.fit(X).predict(X)
runtime = (perf_counter() - start) * 1000
unique = set(labels)
clusters = len(unique - {-1})
noise_points = int(np.sum(labels == -1))
mask = labels != -1
sil = None
if mask.sum() > 2 and len(set(labels[mask])) > 1:
    sil = silhouette_score(X[mask], labels[mask])

a,b,c = st.columns(3)
a.metric("Detected clusters", clusters)
b.metric("Noise points", noise_points)
c.metric("Silhouette", "N/A" if sil is None else f"{sil:.3f}")
st.caption(f"Approximate fit time in this session: {runtime:.1f} ms")

fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1], c=labels, s=28, alpha=.8)
ax.set_title(f"{algorithm} on {dataset}")
ax.set_xlabel("Feature 1 (scaled)")
ax.set_ylabel("Feature 2 (scaled)")
st.pyplot(fig)
plt.close(fig)

st.info("Silhouette score is one diagnostic, not a universal winner. Cluster shape, density, noise and assumptions matter.")
st.markdown("**Skills:** Python · scikit-learn · unsupervised learning · clustering evaluation · visualization")
st.caption("Muhammad Kamil Shah · BS Data Science · Educational portfolio project")
