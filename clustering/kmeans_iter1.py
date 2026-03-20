import math

points = [(2,2), (4,4), (5,5), (6,6), (9,9), (0,4), (4,0)]

centroids = [
    ((2+4+6)/3, (2+4+6)/3),
    ((0+4)/2, (4+0)/2),
    ((5+9)/2, (5+9)/2)
]

clusters = [[], [], []]

for p in points:
    distances = [math.sqrt((p[0]-c[0])**2 + (p[1]-c[1])**2) for c in centroids]
    min_idx = distances.index(min(distances))
    clusters[min_idx].append(p)

new_centroids = []
for cluster in clusters:
    cx = sum(p[0] for p in cluster) / len(cluster)
    cy = sum(p[1] for p in cluster) / len(cluster)
    new_centroids.append((cx, cy))

print("Clusters:", clusters)
print("New Centroids:", new_centroids)
