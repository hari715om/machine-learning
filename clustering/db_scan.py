import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def get_neighbors(X, point_idx, eps):
    distances = np.linalg.norm(X - X[point_idx], axis=1)
    return np.where(distances <= eps)[0].tolist()

def dbscan_custom(X, eps, min_pts):
    labels = np.full(X.shape[0], 0)
    cluster_id = 0

    for p_idx in range(X.shape[0]):
        if labels[p_idx] != 0:
            continue

        neighbors = get_neighbors(X, p_idx, eps)

        if len(neighbors) < min_pts:
            labels[p_idx] = -1
            continue

        cluster_id += 1
        labels[p_idx] = cluster_id

        queue = neighbors.copy()
        queue.remove(p_idx)

        while queue:
            q_idx = queue.pop(0)

            if labels[q_idx] == -1:
                labels[q_idx] = cluster_id

            if labels[q_idx] != 0:
                continue

            labels[q_idx] = cluster_id
            new_neighbors = get_neighbors(X, q_idx, eps)

            if len(new_neighbors) >= min_pts:
                for n in new_neighbors:
                    if labels[n] == 0 or labels[n] == -1:
                        if n not in queue:
                            queue.append(n)

    return labels

X, _ = make_moons(n_samples=500, noise=0.08, random_state=42)

eps_value = 0.12
min_pts_value = 5

custom_labels = dbscan_custom(X, eps=eps_value, min_pts=min_pts_value)

plt.figure(figsize=(10, 6))
unique_labels = set(custom_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]

    class_member_mask = (custom_labels == k)
    xy = X[class_member_mask]

    if k == -1:
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=5)
    else:
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=9)

plt.title(f"DBSCAN Clustering Output (eps={eps_value}, min_pts={min_pts_value})")
plt.xlabel("Feature X1")
plt.ylabel("Feature X2")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()