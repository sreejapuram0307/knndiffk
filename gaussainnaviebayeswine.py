
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(
    y_test, y_pred,
    target_names=data.target_names,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gaussian Naive Bayes Confusion Matrix")
plt.show()

errors = cm.sum(axis=1) - np.diag(cm)

print("Misclassified samples per actual class:", errors)

if errors.sum() == 0:
    print("No sample is misclassified.")
else:
    idx = np.argmax(errors)
    print("Most misclassified actual class:", data.target_names[idx])
    print("Number of errors:", errors[idx])
