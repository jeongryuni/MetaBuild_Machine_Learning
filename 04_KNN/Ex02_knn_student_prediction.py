import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x= np.array([
    [2,9],
    [1,5],
    [3,7],
    [6,2],
    [7,3],
    [8,4]
])

y = np.array([0,0,0,1,1,1])

new_student = np.array([[4,6],[7,2]])

# 모델 학습
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)
pred = model.predict(new_student)
print(f'pred : {pred}')

# 유클리드 거리(Euclidean distance)
# d = √((x2 - x1)² + (y2 - y1)²)
distances, index = model.kneighbors(new_student)
print(f'distance : {distances}')
print(f'index : {index}')

plt.rc('font', family='malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(x[y==0,0], x[y==0,1],  color='red', label = '공부시간', s=100)
plt.scatter(x[y==1,0], x[y==1,1],  color='blue', label = '수면시간', s=100)
plt.scatter(new_student[:,0], new_student[:,1], color='green', label = '새학생', marker='*', s=100)
plt.grid(True)
plt.title('합격/불합격')
plt.show()