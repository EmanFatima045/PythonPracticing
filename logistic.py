
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("housing.csv")

print("===== DATASET PREVIEW =====")
print(df.head())

print("\nShape:", df.shape)

print("\nInfo:")
print(df.info())

print("\nStatistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# NEW: Duplicate check (like your friend)
print("\nDuplicate Rows:")
print(df[df.duplicated(keep=False)])


print("\nCorrelation Matrix:")
print(df.corr())


print("\nCorrelation with House Price:")
print(df.corr()["house_price"].sort_values(ascending=False))


df.hist(figsize=(12,10))
plt.suptitle("Feature Distributions")
plt.show()

plt.scatter(df['rooms'], df['house_price'])
plt.xlabel("Rooms")
plt.ylabel("Price")
plt.title("Rooms vs Price")
plt.show()


for col in df.columns:
    if col != "house_price":
        plt.scatter(df[col], df["house_price"])
        plt.xlabel(col)
        plt.ylabel("house_price")
        plt.title(f"{col} vs Price")
        plt.show()


df.boxplot(figsize=(12,6))
plt.title("Outlier Detection")
plt.show()

Q1 = df['house_price'].quantile(0.25)
Q3 = df['house_price'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['house_price'] < lower) | (df['house_price'] > upper)]

print("\nQ1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower)
print("Upper Bound:", upper)

print("\nOutliers Found:", len(outliers))
print(outliers)

df['price_per_room'] = df['house_price'] / df['rooms']

print("\nTop Properties by Value (Low Price per Room = Better):")
print(df.sort_values("price_per_room")[["house_price", "rooms", "price_per_room"]].head())

X = df.drop("house_price", axis=1)
y = df["house_price"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled Feature Sample:")
print(X_scaled.head())

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print("R2 Score:", score)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()