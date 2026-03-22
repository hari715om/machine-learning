import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.5, random_state=42)

linkage_methods = ['single', 'complete', 'average', 'centroid']

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
axes1 = axes1.flatten()

for i, method in enumerate(linkage_methods):
    Z = linkage(X, method=method)
    
    dendrogram(Z, ax=axes1[i], truncate_mode='lastp', p=12, show_leaf_counts=True)
    
    axes1[i].set_title(f"Dendrogram: {method.capitalize()} Linkage")
    axes1[i].set_xlabel("Number of points in node")
    axes1[i].set_ylabel("Merge Distance")

plt.tight_layout()
plt.show()

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2 = axes2.flatten()

for i, method in enumerate(linkage_methods):
    Z = linkage(X, method=method)
    
    labels = fcluster(Z, t=4, criterion='maxclust')
    
    for cluster_id in np.unique(labels):
        mask = (labels == cluster_id)
        axes2[i].scatter(X[mask, 0], X[mask, 1], s=60, edgecolor='black', alpha=0.8)
        
    axes2[i].set_title(f"Clusters: {method.capitalize()} Linkage")
    axes2[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()