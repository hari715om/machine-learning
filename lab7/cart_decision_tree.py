import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = {
    'Income': [60, 75, 85.5, 52.8, 64.8, 64.8, 61.5, 43.2, 87, 84, 
               110.1, 49.2, 108, 59.2, 82.8, 66, 69, 47.4, 93, 33, 
               51, 51, 81, 63],
    'Lot_Size': [18.4, 19.6, 16.8, 20.8, 21.6, 17.2, 20.8, 20.4, 23.6, 17.6, 
                 19.2, 17.6, 17.6, 16, 22.4, 18.4, 20, 16.4, 20.8, 18.8, 
                 22, 14, 20, 14.8],
    'Ownership': ['Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner',
                  'Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner', 'Owner', 'Nonowner',
                  'Owner', 'Nonowner', 'Owner', 'Nonowner']
}

df = pd.DataFrame(data)

X = df[['Income', 'Lot_Size']]
y = df['Ownership']

model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X, y)

new_household = [[60, 20]]
prediction = model.predict(new_household)
print(f"Prediction for Income=60, Lot_Size=20: {prediction[0]}")

plt.figure(figsize=(12, 8))
plot_tree(model, 
          feature_names=['Income', 'Lot_Size'], 
          class_names=model.classes_, 
          filled=True, 
          rounded=True)
plt.title("CART Decision Tree: Lawn Tractor Ownership")
plt.show()