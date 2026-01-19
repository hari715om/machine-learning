x = list(map(float, input("Enter x values: ").split()))
y = list(map(float, input("Enter y values: ").split()))

n = len(x)

sorted_x = sorted(x)

if n % 2 == 1:  
    median_val = sorted_x[n // 2]
else:           
    median_val = (sorted_x[n//2 - 1] + sorted_x[n//2]) / 2

index_map = {sorted_x[i]: i - (n // 2) for i in range(n)}

if n % 2 == 0:
    for i in range(n):
        index_map[sorted_x[i]] = i - (n/2 - 0.5)

x_norm = [index_map[xi] for xi in x]

m = (n * sum(x_norm[i] * y[i] for i in range(n)) - sum(x_norm) * sum(y)) / \
    (n * sum(i*i for i in x_norm) - (sum(x_norm) ** 2))

b_norm = (sum(y) - m * sum(x_norm)) / n

print("\nIndex-normalized x values:", x_norm)
print("Slope (m):", m)
print("Intercept (b):", b_norm)
print(f"Normalized-line eq: y = {m:.4f} * x + {b_norm:.4f}")
