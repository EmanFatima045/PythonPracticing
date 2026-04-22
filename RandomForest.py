
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("iris.csv")

print("DATASET LOADED ")
print(df.head())

print("\n DATASET INFO")
df.info()

print("\n MISSING VALUES")
print(df.isnull().sum())

X = df.drop("species", axis=1)
y = df["species"]

print("\nFEATURES")
print(X.head())

print("\n TARGET ")
print(y.head())

# 5. SPLIT DATA INTO TRAIN & TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
print("\n===== MODEL TRAINED SUCCESSFULLY =====")

y_pred = model.predict(X_test)

print("\n===== PREDICTIONS =====")
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)


print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

sample = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = model.predict(sample)

print("Sample Input:", sample)
print("Predicted Class:", prediction[0])