pip install xgboost
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

models = {
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=0),
    "XGBoost": XGBClassifier(
        n_estimators=100, eval_metric="mlogloss", random_state=0
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=0
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

print("Model\t\t\tAccuracy")

for name, accuracy in results.items():
    print(name, "\t", accuracy)

best_model = max(results, key=results.get)
print("Best model by accuracy:", best_model)