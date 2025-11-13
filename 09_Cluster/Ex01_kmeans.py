import numpy as np
from sklearn.cluster import KMeans

x = np.array([
    [1,2],
    [1,4],
    [1,0],
    [10,2],
    [10,4],
    [10,0]
])
# K-means
# 데이터를 K개의 군집으로 나누어, 각 군집의 중심(centroid)과의 제곱거리 합(=inertia)을 최소화
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(x)

print("라벨:", kmeans.labels_)            # 각 점의 군집 번호(순서는 임의)
print("중심:", kmeans.cluster_centers_)   # (2,2)와 (10,2) 부근으로 수렴할 것 (중심좌표)
print("관성(inertia):", kmeans.inertia_)  # 군집 내 제곱거리 합

# 중심좌표와의 거리
distances = kmeans.transform(x)
print('distances:\n', distances)

# 새로운 데이터 예측
new_data = np.array([
    [3,3],
    [7,2]
])

new_labels = kmeans.predict(new_data)
print('new_labels:\n', new_labels)

distances = kmeans.transform(new_data)
print('distances:\n', distances)