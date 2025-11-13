import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../00_dataIn/Wholesale customers data.csv')
df = df.iloc[:,2:]

# 기본 통계 요약
#print(df.describe())

# 표준화
scaler = StandardScaler()
x = scaler.fit_transform(df)

# n_clusters=5 → 5개의 그룹으로 나누기
# n_init=10 → 초기 중심점을 10번 바꿔 시도하여 최적 결과 선택
kmeans = KMeans(n_clusters=5, n_init=10, random_state=1234)
kmeans.fit(x)

# 각 데이터가 속한 클러스터 번호 출력
print(kmeans.labels_)
# 각 군집별 중심좌표
print(kmeans.cluster_centers_)

df['Cluster'] = kmeans.labels_
print(df.head(5))

# 그룹별 빈도수
print(df['Cluster'].value_counts())

plt.figure(figsize=(8,6))

colors = ['red', 'blue', 'green', 'orange', 'purple']
for i in range(5):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(
        cluster_data['Milk'], cluster_data['Grocery'], color=colors[i], label=f'Cluster {i}')

plt.xlim(0, 13000)   # x축 0 ~ 4만
plt.ylim(0, 40000)   # y축 0 ~ 4만
plt.legend()
plt.xlabel('Milk')
plt.ylabel('Grocery')
plt.show()