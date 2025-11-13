import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# DBSCAN : 밀도기반 군집분석 방법
x = np.array([
    [1,2], [2,2], [2,3], # 군집1
    [8,7], [8,8], [7,8], # 군잡2
    [20,30] # 노이즈(잡음) 처리
])

# 반경 0.5, 1.5, 3.0에 있으면 이웃
eps_values = [0.5, 1.5, 3.0]

# 반경안에 최소 2개는 있어야함
min_samples = 2

for i, eps in enumerate(eps_values):
    # print(i, eps)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(x)
    labels = dbscan.fit_predict(x)
    print(f'labels : {labels}') # -1 노이즈처리/ 0 그룹/ 1 그룹

    plt.subplot(1, len(eps_values), i + 1)
    plt.title(f'DBSCAN eps = {eps}')

    # set 첫번쨰라벨 -1/ 두번쨰라벨 -1, 0, 1 / 세번쨰라벨 0, 1 -1
    for label in set(labels):
        color = 'blue' if label == 1 else 'red'
        marker = 'x' if label == -1 else 'o'
        label_name = 'Noise' if label == -1 else f'Cluster {label}'

        cluster_points = x[labels == label]

        plt.scatter(cluster_points[:,0], cluster_points[:,1], c=color, marker=marker, label=label_name)
        plt.legend()
plt.show()


# eps_values = [0.5, 1.5, 3.0] # 반경
#
# plt.figure(figsize=(15,5))
# for i, eps in enumerate(eps_values):
#     dbscan = DBSCAN(eps=eps, min_samples=2)
#     labels = dbscan.fit_predict(x)
#
#     # 군집이 아닌 노이즈(-1) 분리
#     mask_noise = labels == -1
#     mask_cluster = labels != -1
#
#     plt.subplot(1,3,i+1)
#     # 군집 점들
#     plt.scatter(x[mask_cluster,0], x[mask_cluster,1],
#                 c=labels[mask_cluster], cmap='rainbow', s=100, label='Clusters')
#     # 노이즈
#     plt.scatter(x[mask_noise,0], x[mask_noise,1],
#                 c='k', marker='x', s=120, label='Noise')
#
#     plt.title(f'eps={eps}')
#     plt.legend()
#
# plt.show()