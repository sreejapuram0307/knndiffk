
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_iris()
X = data.data
y = data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 3. Train k-NN for different k values
k_values = [1, 3, 5, 7, 9]
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print("k =", k, "Accuracy =", acc)

# 4. Find best k (ties choose smallest k)
best_index = accuracies.index(max(accuracies))
print("Best k =", k_values[best_index])

# 5. Plot accuracy versus k
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("k-NN Accuracy versus k")
plt.grid()
plt.show()
