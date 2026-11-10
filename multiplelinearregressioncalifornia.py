
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = fetch_california_housing()
X = data.data
y = data.target

print("Features:", data.feature_names)
print("Target: Median house value (in units of $100,000)")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual versus Predicted House Prices")

low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())

plt.plot([low, high], [low, high], "r--")
plt.show()


for name, coef in zip(data.feature_names, model.coef_):
    print(name, ":", coef)

idx = np.argmax(np.abs(model.coef_))
print("Most influential feature:", data.feature_names[idx])
