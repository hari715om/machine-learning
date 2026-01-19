from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Manual confusion matrix
num_classes = 3
conf_matrix = [[0]*num_classes for _ in range(num_classes)]

for actual, pred in zip(y_test, y_pred):
    conf_matrix[actual][pred] += 1

print("Confusion Matrix:")
for row in conf_matrix:
    print(row)

# Metrics
total_samples = sum(sum(row) for row in conf_matrix)

precision_list, recall_list, f1_list, support_list = [], [], [], []

for i in range(num_classes):
    TP = conf_matrix[i][i]
    FN = sum(conf_matrix[i]) - TP
    FP = sum(conf_matrix[j][i] for j in range(num_classes)) - TP
    TN = total_samples - (TP + FP + FN)

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0

    support = sum(conf_matrix[i])

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    support_list.append(support)

# Weighted averages
weighted_precision = sum(p*s for p,s in zip(precision_list, support_list)) / total_samples
weighted_recall = sum(r*s for r,s in zip(recall_list, support_list)) / total_samples
weighted_f1 = sum(f*s for f,s in zip(f1_list, support_list)) / total_samples

print("\nWeighted Precision:", weighted_precision)
print("Weighted Recall:", weighted_recall)
print("Weighted F1-score:", weighted_f1)
