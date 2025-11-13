from cProfile import label

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    'Action': [9, 8, 2, 1, 7, 6, 3, 2, 9, 8],
    'Romance':[1, 2, 8, 9, 3, 2, 7, 9, 1, 3],
    'Horror':[8, 7, 2, 1, 7, 6, 3, 2, 8, 7],
    'Comedy': [4, 5, 7, 8, 5, 4, 6, 7, 3, 4],
    'Drama': [2, 3, 9, 8, 4, 3, 8, 9, 2, 3]
     }
)

scaler = StandardScaler()
scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3)
kmeans.fit(scaled)
print(kmeans.labels_)
distances = kmeans.transform(scaled)

data['Cluster'] = kmeans.labels_

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(7, 5))
plt.scatter(data['Action'], data['Romance'], c=data['Cluster'], s=130)
for i in range(10):
    plt.text(data['Action'][i]+0.1, data['Romance'][i]+0.1,  f"P{i+1}", color='black')

plt.title('K-Means 클러스터링 (Action vs Romance)')
plt.xlabel('Action')
plt.ylabel('Romance')
plt.show()
