
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names) # type: ignore

X = df[['petal length (cm)', 'petal width (cm)']]

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

df['Cluster'] = kmeans.labels_

plt.scatter(X['petal length (cm)'], X['petal width (cm)'], c=df['Cluster'])

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200)

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Simple K-Means Clustering')
plt.show()

print(df.head())