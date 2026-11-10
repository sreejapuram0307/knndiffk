from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]


model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="tanh",
    solver="lbfgs",
    max_iter=1000,
    random_state=0
)

model.fit(X, y)


y_pred = model.predict(X)


print("Input\tActual\tPredicted")
for x, actual, pred in zip(X, y, y_pred):
    print(x, "\t", actual, "\t", pred)

print("MLP Accuracy:", accuracy_score(y, y_pred))

single = Perceptron(max_iter=1000, random_state=0)
single.fit(X, y)

single_pred = single.predict(X)

print("Single-layer predictions:", single_pred)
print("Single-layer accuracy:", accuracy_score(y, single_pred))