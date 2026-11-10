
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

for criterion in ["entropy", "gini"]:
    model = DecisionTreeClassifier(criterion=criterion, random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nCriterion:", criterion)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="macro"))
    print("Recall:", recall_score(y_test, y_pred, average="macro"))
    print("F1-score:", f1_score(y_test, y_pred, average="macro"))
