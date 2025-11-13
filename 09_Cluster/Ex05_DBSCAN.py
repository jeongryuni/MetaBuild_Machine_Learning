import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('../00_dataIn/middle_shcool_graduates_report.xlsx')

dummy_list = ['지역', '코드', '유형', '주야']
for i in dummy_list:
    print(f'{i} : {df[i].unique()}') #값 목록
    print(f'{i} : {df[i].nunique()}') #고유값 개수
    print('--------------------------')

print('범주형 데이터의 원-핫 인코딩')
df_encoded = pd.get_dummies(df, columns=dummy_list, dtype=int)
print(df_encoded.head(10))
print(df_encoded.columns)

train_feature = ['과학고', '외고_국제고', '자사고', '자공고', '유형_공립', '유형_국립', '유형_사립']
x = df_encoded[train_feature]
print(f'표준화 전 x : {x}')

# 표준화
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
print(f'표준화 전 x : {x}')

# 모델
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan.fit(x)
cluster_label = dbscan.labels_ # 어느 군집(Cluster)에 속하는지

# return_counts=True 개수까지 출력
unique_vals, counts = np.unique(cluster_label, return_counts=True)
print(f'unique vals : {unique_vals}') # 라벨 : [-1  0  1  2  3]
print(f'counts : {counts}') # 라벨 개수 : [ 92 223  90   5   5]

# df_encoded : 원본 칼럼 + 원핫인코딩
df_encoded['Cluster'] = dbscan.labels_
print(df_encoded.shape)

# 클러스터별 빈도수
print(df_encoded['Cluster'].value_counts())

# 클러스터별 그룹핑
concern = ['학교명', '과학고', '외고_국제고', '자사고']

# 군집 번호
grouped = df_encoded.groupby('Cluster')
for group_no, group in grouped:
    print('군집 번호:', group_no)
    print(f'군집별 데이터 개수 : {len(group)}')
    print(group.loc[:,concern])
    print('----------------------')


import folium
from folium.plugins import MarkerCluster

# colors = {
#     -1: 'gray',
#     0: 'orange',
#     1: 'blue',
#     2: 'green',
#     3: 'purple'
# }
school = df_encoded['학교명']
x = df_encoded['위도']
y = df_encoded['경도']
# m = folium.Map(location=[37.658978, 127.181755], zoom_start=30)

# for i in range(df_encoded.shape[0]):
#     folium.Marker(
#         location=[x.iloc[i], y.iloc[i]],
#         popup=school.iloc[i],
#         icon=folium.Icon(color=colors[df_encoded.iloc[i]['Cluster']])
#     ).add_to(m)
#
# m.save('../00_dataIn/schoolMap2.html')



colors = {-1: 'gray', 0: 'red', 1: 'green', 2: 'pink', 3: 'lightblue'}
m = folium.Map(location=[37.487841, 127.039141], zoom_start=12)
folium.Marker(location=[37.487841, 127.039141],
              popup="home",
              icon=folium.Icon(color='black', icon='home')).add_to(m)

for cluster_num in df_encoded['Cluster'].unique():
    df_cluster = df.loc[df_encoded['Cluster'] == cluster_num, ['학교명', '위도', '경도']]
    for name, lat, lon in zip(df_cluster['학교명'], df_cluster['위도'], df_cluster['경도']):
        folium.Marker(
            location=[lat, lon],
            popup=name,
            icon=folium.Icon(color=colors.get(cluster_num))
        ).add_to(m)
m.save('school.html')