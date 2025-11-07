import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

plt.rc('font', family='malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

x = np.array([
    [1,2],
    [2,3],
    [1,3],#사과
    [8,8],
    [9,7],
    [8,9] #바나나
])

y = np.array([0,0,0,1,1,1])

new_fruit = np.array([[2,2], [9,8]])

model = RandomForestClassifier(n_estimators=100)
model.fit(x, y)
pred = model.predict(new_fruit)
print(pred)


# 사과 : 빨간점
# 바나나 : 노란점
# 새과일 : 초록색

# K-최근접 이웃 알고리즘
# 가장 가까운 이웃 1개만 참고해서 분류 (n_neighbors = 이웃개수)
model = KNeighborsClassifier(n_neighbors=1)


# 기존점 클래스 거리계산 (유클리드 거리 계싼)
# d = √((x2 - x1)² + (y2 - y1)²)
# 새 과일 [2,2] 기준으로 기존 점들과의 거리 계산 예시
# [1,2]  → √((2-1)² + (2-2)²) = √(1 + 0) = 1.00
# [2,3]  → √((2-2)² + (2-3)²) = √(0 + 1) = 1.00
# [1,3]  → √((2-1)² + (2-3)²) = √(1 + 1) = 1.41
# [8,8]  → √((2-8)² + (2-8)²) = √(36 + 36) = 8.49
# [9,7]  → √((2-9)² + (2-7)²) = √(49 + 25) = 8.60
# [8,9]  → √((2-8)² + (2-9)²) = √(36 + 49) = 9.22
# 가장 가까운 3개 → [1,2], [2,3], [1,3] → 모두 사과(0)
# → 새 과일 [2,2]의 예측 클래스: 사과(0)

model.fit(x, y)
pred = model.predict(new_fruit)
print(pred)


plt.scatter(x[y==0, 0], x[y==0, 1], color='r', label='사과')
plt.scatter(x[y==1, 0], x[y==1, 0], color='y', label='바나나')
plt.scatter(new_fruit[:,0], new_fruit[:,1], color='g', label='새과일',s=270, marker='*')
plt.legend(loc='lower right')
plt.title('KNN 예제')
plt.xlabel('색상')
plt.ylabel('무게')
plt.show()