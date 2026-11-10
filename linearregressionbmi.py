
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
data = load_diabetes()
X = data.data[:, [2]]  # BMI feature, keep 2D shape
y = data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 3. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluation metrics
mse = mean_squared_error(y_test, y_pred)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("R2 Score:", r2_score(y_test, y_pred))

# 6. Plot actual and predicted values
order = np.argsort(X_test[:, 0])

plt.scatter(X_test[:, 0], y_test, label="Actual")
plt.plot(
    X_test[order, 0],
    y_pred[order],
    color="red",
    label="Predicted"
)

plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("BMI versus Diabetes Progression")
plt.legend()
plt.show()

# 7. Predict for BMI = 0.05
result = model.predict([[0.05]])
print("Prediction for BMI 0.05:", result[0])
