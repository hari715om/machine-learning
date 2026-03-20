import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Mall_Customers.csv")

X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cov_matrix = np.cov(X_scaled.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)

print("Eigenvalues:\n", sorted_eigenvalues)
print("\nEigenvectors:\n", sorted_eigenvectors)
print("\nExplained Variance Ratio:\n", explained_variance)

n_components = 2
eigenvector_subset = sorted_eigenvectors[:, 0:n_components]

X_reduced = np.dot(X_scaled, eigenvector_subset)

plt.figure(figsize=(8,6))
plt.scatter(X_reduced[:,0], X_reduced[:,1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Mall Customers Dataset")
plt.show()