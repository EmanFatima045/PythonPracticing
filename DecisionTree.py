import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score
# Load dataset
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

# Duplicate check
print("\nDuplicate Rows:")
print(df[df.duplicated(keep=False)])


# Histograms
df.hist(figsize=(12,10))
plt.suptitle("Feature Distributions")
plt.show()



# Outlier Detection using IQR
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

# Feature Engineering
df['price_per_room'] = df['house_price'] / df['rooms']

print("\nTop Properties by Value (Low Price per Room = Better):")
print(df.sort_values("price_per_room")[["house_price", "rooms", "price_per_room"]].head())

# Define features and target
X = df.drop("house_price", axis=1)
y = df["house_price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree Model
model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


# Actual vs Predicted Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()

# Visualize Decision Tree
plt.figure(figsize=(15,10))
plot_tree(model, feature_names=X.columns, filled=True)  #type: ignore
plt.title("Decision Tree Visualization")
plt.show()