import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for i in range(n_samples):
                condition = y[i] * (np.dot(X[i], self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y[i] * X[i])
                    self.b -= self.lr * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)


X, y = make_blobs(n_samples=200, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)

model = LinearSVM(n_iters=2000)
model.fit(X, y)

def plot_linear_svm(X, y, model):
    plt.figure(figsize=(7,5))
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.c_[XX.ravel(), YY.ravel()]
    Z = (np.dot(xy, model.w) - model.b).reshape(XX.shape)

    ax.contour(XX, YY, Z, levels=[-1,0,1], colors='k', linestyles=['--','-','--'])
    plt.title("Linear SVM")
    plt.show()

plot_linear_svm(X, y, model)