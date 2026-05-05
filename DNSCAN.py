# =========================================================
# ADVANCED FRAUD DETECTION (CLEAN VERSION)
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score
)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("creditcard.csv").drop_duplicates()

X = df.drop("Class", axis=1)
y = df["Class"]

print("Dataset:", df.shape)
print("Fraud %:", round(y.mean()*100, 4))

# ---------------------------------------------------------
# 2. SCALE DATA
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 3. PCA (Dimensionality Reduction)
# ---------------------------------------------------------
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance:", round(sum(pca.explained_variance_ratio_)*100, 2), "%")

# Plot PCA variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100, marker='o')
plt.title("PCA Explained Variance")
plt.xlabel("Components")
plt.ylabel("Variance %")
plt.grid()
plt.show()

# ---------------------------------------------------------
# 4. SAMPLING (for speed)
# ---------------------------------------------------------
np.random.seed(42)
idx = np.random.choice(len(X_pca), 15000, replace=False)
X_sample = X_pca[idx]
y_sample = y.iloc[idx]

# ---------------------------------------------------------
# 5. DBSCAN (Improved)
# ---------------------------------------------------------
model = DBSCAN(eps=2.5, min_samples=8)
clusters = model.fit_predict(X_sample)

# -1 = anomaly
y_pred = (clusters == -1).astype(int)

print("Clusters found:", len(set(clusters)))
print("Fraud predicted:", y_pred.sum())

# ---------------------------------------------------------
# 6. METRICS
# ---------------------------------------------------------
print("\nClassification Report:\n")
print(classification_report(y_sample, y_pred))

roc = roc_auc_score(y_sample, y_pred)
f1  = f1_score(y_sample, y_pred)

print("ROC-AUC:", roc)
print("F1 Score:", f1)

cm = confusion_matrix(y_sample, y_pred)

# ---------------------------------------------------------
# 7. VISUALIZATION
# ---------------------------------------------------------

# 🔹 Confusion Matrix
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 🔹 ROC Curve
fpr, tpr, _ = roc_curve(y_sample, y_pred)

plt.figure()
plt.plot(fpr, tpr, label="DBSCAN")
plt.plot([0,1], [0,1], '--')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()


# 🔹 Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_sample, y_pred)

plt.figure()
plt.plot(rec, prec)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# 🔹 PCA Scatter (Actual)
plt.figure()
plt.scatter(X_sample[y_sample==0,0], X_sample[y_sample==0,1], s=2, label="Legit")
plt.scatter(X_sample[y_sample==1,0], X_sample[y_sample==1,1], s=10, label="Fraud")
plt.legend()
plt.title("Actual Data (PCA)")
plt.show()


# 🔹 PCA Scatter (Predicted)
plt.figure()
plt.scatter(X_sample[y_pred==0,0], X_sample[y_pred==0,1], s=2, label="Predicted Legit")
plt.scatter(X_sample[y_pred==1,0], X_sample[y_pred==1,1], s=10, label="Predicted Fraud")
plt.legend()
plt.title("DBSCAN Prediction")
plt.show()