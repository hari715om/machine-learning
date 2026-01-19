import numpy as np
import pandas as pd

# Data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([5, 6, 10, 13, 11], dtype=float)

# Initial values
beta0_init = 3.3
beta1_init = 1.9
lr = 0.1

# ---------------- Batch Gradient Descent ----------------
def batch_gradient_descent_table(x, y, beta0, beta1, lr, epochs=3):
    n = len(x)
    rows = []

    for epoch in range(1, epochs+1):
        y_pred = beta0 + beta1 * x
        error = y_pred - y

        db0 = (1/n) * np.sum(error)
        db1 = (1/n) * np.sum(error * x)

        beta0_new = beta0 - lr * db0
        beta1_new = beta1 - lr * db1

        rows.append([epoch, np.round(db0,4), np.round(db1,4),
                     np.round(beta0_new,4), np.round(beta1_new,4)])

        beta0, beta1 = beta0_new, beta1_new

    df = pd.DataFrame(rows, columns=["Epoch", "dBeta0", "dBeta1", "Beta0", "Beta1"])
    return df


# ---------------- Stochastic Gradient Descent ----------------
def stochastic_gradient_descent_table(x, y, beta0, beta1, lr, epochs=2):
    rows = []

    for epoch in range(1, epochs+1):
        for i in range(len(x)):
            y_pred = beta0 + beta1 * x[i]
            error = y_pred - y[i]

            db0 = error
            db1 = error * x[i]

            beta0_new = beta0 - lr * db0
            beta1_new = beta1 - lr * db1

            rows.append([epoch, x[i], y_pred, error,
                         np.round(beta0_new,4), np.round(beta1_new,4)])

            beta0, beta1 = beta0_new, beta1_new

    df = pd.DataFrame(rows, columns=["Epoch", "x", "y_pred", "Error", "Beta0", "Beta1"])
    return df


# Run BGD
print("\n===== Batch Gradient Descent Table =====")
df_bgd = batch_gradient_descent_table(x, y, beta0_init, beta1_init, lr)
print(df_bgd.to_string(index=False))

# Run SGD
print("\n===== Stochastic Gradient Descent Table =====")
df_sgd = stochastic_gradient_descent_table(x, y, beta0_init, beta1_init, lr)
print(df_sgd.to_string(index=False))
