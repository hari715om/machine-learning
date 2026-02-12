import numpy as np

# Data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([5, 6, 10, 13, 11], dtype=float)

# Initial parameters
beta0 = 3.3
beta1 = 1.9
lr = 0.1   # learning rate

def batch_gradient_descent(x, y, beta0, beta1, lr, epochs=10):
    n = len(x)

    for _ in range(epochs):
        # Predictions
        y_pred = beta0 + beta1 * x

        # Errors
        error = y_pred - y

        # Gradients (mean of all samples)
        db0 = (1/n) * np.sum(error)
        db1 = (1/n) * np.sum(error * x)

        # Parameter update
        beta0 -= lr * db0
        beta1 -= lr * db1

    return beta0, beta1


# ---------- Stochastic Gradient Descent ----------
def stochastic_gradient_descent(x, y, beta0, beta1, lr, epochs=10):
    n = len(x)

    for _ in range(epochs):
        for i in range(n):  # update for each sample
            y_pred = beta0 + beta1 * x[i]
            error = y_pred - y[i]

            # Gradients for one sample
            db0 = error
            db1 = error * x[i]

            # Update
            beta0 -= lr * db0
            beta1 -= lr * db1

    return beta0, beta1


# Run Batch GD
b0_bgd, b1_bgd = batch_gradient_descent(x, y, beta0, beta1, lr)
print("After Batch Gradient Descent:")
print("beta0 =", b0_bgd, "beta1 =", b1_bgd)

# Reset initial values
beta0 = 3.3
beta1 = 0.1

# Run SGD
b0_sgd, b1_sgd = stochastic_gradient_descent(x, y, beta0, beta1, lr)
print("\nAfter Stochastic Gradient Descent:")
print("beta0 =", b0_sgd, "beta1 =", b1_sgd)
