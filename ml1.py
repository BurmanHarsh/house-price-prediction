import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("train.csv")


features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

data = df[features + [target]]


data = data.dropna()


X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print("Model Coefficients:", model.coef_)
# print("Model Intercept:", model.intercept_)
# print("Mean Squared Error (MSE):", mse)
# print("R-squared (R² Score):", r2)

# Show model results with explanations
print("\n📊 Linear Regression Model Summary:")
print("--------------------------------------------------")
print(f"🏠 Coefficient for 'GrLivArea' (Living Area): ₹{model.coef_[0]:,.2f} per sq ft")
print(f"🛏️ Coefficient for 'BedroomAbvGr' (Bedrooms Above Ground): ₹{model.coef_[1]:,.2f} per bedroom")
print(f"🛁 Coefficient for 'FullBath' (Full Bathrooms): ₹{model.coef_[2]:,.2f} per bathroom")
print(f"\n🧮 Intercept (Base Price when all features are 0): ₹{model.intercept_:,.2f}")
print("--------------------------------------------------")
print(f"📉 Mean Squared Error (Average squared prediction error): {mse:,.2f}")
print(f"📈 R² Score (Accuracy of model): {r2 * 100:.2f}%")
print("--------------------------------------------------")



plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # identity line
plt.show()
