import math

data=[
    (25, 40000, 'N'),
    (35, 60000, 'N'),
    (45, 80000, 'N'),
    (20, 20000, 'N'),
    (35, 120000, 'N'),
    (53, 18000, 'N'),
    (23, 95000, 'Y'),
    (40, 62000, 'Y'),
    (60, 100000, 'Y'),
    (33, 150000, 'Y'),
]

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def knn_predict(age, loan, k=3):
    distances = []

    for record in data:
        dist = euclidean_distance(age, loan, record[0], record[1])
        distances.append((dist, record[2]))

    distances.sort(key=lambda x: x[0])

    nearest = distances[:k]

    yes = 0
    no = 0
    for item in nearest:
        if item[1] == 'Y':
            yes += 1
        else:
            no += 1

    return 'Y' if yes > no else 'N'

age = int(input("Enter Age: "))
loan = int(input("Enter Loan Amount: "))

result = knn_predict(age, loan)

if result == 'Y':
    print("Default pred: Y")
else:
    print("Default pred: N")
