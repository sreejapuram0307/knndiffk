import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

kernels = ["linear", "poly", "rbf"]
wine_scores = {}

for kernel in kernels:
    model = make_pipeline(
        StandardScaler(), SVC(kernel=kernel)
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    wine_scores[kernel] = accuracy_score(y_test, pred)

print("Wine Dataset Accuracy")
for kernel, score in wine_scores.items():
    print(kernel, ":", score)

X, y = make_moons(n_samples=300, noise=0.2, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300)
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, kernel in zip(axes, kernels):
    model = make_pipeline(
        StandardScaler(), SVC(kernel=kernel)
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("Moons", kernel, "Accuracy:", accuracy_score(y_test, pred))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="black")
    ax.set_title(kernel + " kernel")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()