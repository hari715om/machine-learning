import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.6, random_state=42)

clusters = [X]
target_clusters = 4

while len(clusters) < target_clusters:
    largest_cluster_idx = -1
    max_variance = -1
    
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            variance = np.var(cluster)
            if variance > max_variance:
                max_variance = variance
                largest_cluster_idx = i
                
    cluster_to_split = clusters.pop(largest_cluster_idx)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    split_labels = kmeans.fit_predict(cluster_to_split)
    
    cluster_1 = cluster_to_split[split_labels == 0]
    cluster_2 = cluster_to_split[split_labels == 1]
    
    clusters.append(cluster_1)
    clusters.append(cluster_2)

plt.figure(figsize=(10, 6))
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(clusters))]

for i, (cluster, col) in enumerate(zip(clusters, colors)):
    plt.scatter(cluster[:, 0], cluster[:, 1], s=60, color=col, edgecolor='black', 
                alpha=0.8, label=f'Cluster {i+1}')

plt.title("Divisive Hierarchical Clustering (Top-Down Bisecting)")
plt.xlabel("Feature X1")
plt.ylabel("Feature X2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()