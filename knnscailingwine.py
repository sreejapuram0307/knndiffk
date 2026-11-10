
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_wine()
X = data.data
y = data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 3. k-NN without scaling
model1 = KNeighborsClassifier(n_neighbors=5)
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, pred1)

# 4. Apply scaling: fit only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. k-NN with scaling
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train_scaled, y_train)

pred2 = model2.predict(X_test_scaled)
acc2 = accuracy_score(y_test, pred2)

# 6. Compare results
print("Accuracy without scaling:", acc1)
print("Accuracy with scaling:", acc2)

plt.bar(
    ["Without Scaling", "With Scaling"],
    [acc1, acc2]
)
plt.ylabel("Accuracy")
plt.title("Effect of Feature Scaling on k-NN")
plt.show()
