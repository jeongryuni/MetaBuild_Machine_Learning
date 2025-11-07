import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.sparse import vstack
from sklearn.neighbors import KNeighborsClassifier

x = np.array([
    [1,2],
    [2,3],
    [2,2.5],
    [3,4],
    [4,5],
    [7,5],
    [8,8],
    [9,7]
])

y = np.array([0,0,0,0,0,1,1,1])

# 이웃 개수
k_values = [1,3,5]

# 새로운 이웃
new_point = np.array([[6,5]])

plt.rc('font', family='malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15,5))
for i, k in enumerate(k_values):
    # 모델 생성 및 학습
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x, y)

    # 새 데이터 예측
    pred = model.predict(new_point)
    print(f'k = {k}, pred : {pred}')

    #  새 데이터와 기존 데이터 간 거리 계산
    distances, index = model.kneighbors(new_point)
    print(f'distance : {distances}')
    print(f'index : {index}')
    print()

    # 결정 경계(Decision Boundary) 계산
    # x, y 축 범위 지정 (그래프 영역 설정)
    x_min, x_max = x[:, 0].min(), x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min(), x[:, 1].max() + 1

    # 일정 간격(0.1)으로 meshgrid 생성
    # meshgrid()는 2차원 평면 위의 점들을 격자 형태(grid) 만들어줌
    # → 모든 점에서의 예측값을 계산하기 위해 사용
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

    # meshgrid 좌표를 하나로 합쳐서 (x,y) 쌍으로 변환
    # 1차원 배열을 세로 방향으로 쌓고, .T를 적용하면 행과 열을 뒤바꿈
    xy = (np.vstack((xx.ravel(), yy.ravel()))).T

    # # 각 좌표의 예측 결과 (0 or 1)
    z = model.predict(xy)
    z = z.reshape(xx.shape)  # 그래프 표시를 위해 원래 형태로 재구성
    print(z)

    # 시각화
    plt.subplot(1, 3, i+1)
    plt.scatter(x[y == 0, 0], x[y == 0, 1], c='r', s=100, edgecolors='k', label='사과' , zorder=10)
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='y', s=100, edgecolors='k', label='바나나', zorder=10)
    color = 'b' if pred == 0 else 'g'
    plt.scatter(new_point[0, 0], new_point[0, 1], c=color, s=100, marker='X', edgecolors='k', label='새과일', zorder=3)

    plt.title(f'사과 바나나 예측 k = {k}', fontsize=13)
    plt.contourf(xx, yy, z, colors=['#FADADD', '#D6EAF8'], levels=[-1,0,1], alpha=0.8)
    plt.contour(xx, yy, z, colors=['red'], levels=[0], alpha=0.3)
    plt.xlabel('색상')
    plt.ylabel('무게')
    plt.legend(loc='upper left')
    plt.grid(True)

plt.show()