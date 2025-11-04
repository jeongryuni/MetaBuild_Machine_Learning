# 시험점수 (국어, 영어) => 합격/불합격
import numpy as np
from matplotlib import pyplot as plt
from numpy.ma.extras import vstack
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 시험점수
np.random.seed(111)
# 합격
X_pass = np.random.randn(20, 2) *5 + [70,75]

# 불합격
X_fail = np.random.randn(20, 2) *5 + [50,55]

# 두 배열 합치기
x = vstack((X_pass, X_fail))
y = np.array([1]*20 + [0]*20)

# 학습/테스트용 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# SVC모델 학습
model = SVC(kernel='rbf')
model.fit(x_train, y_train)
print(model.intercept_)

# 그래프
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8,6))
plt.scatter(x_train[y_train==1,0], x_train[y_train==1,1], color='blue', label='합격')
plt.scatter(x_train[y_train==0,0], x_train[y_train==0,1], color='red', label='불합격')

ax = plt.gca()
# x축 y축 크기
xlim = ax.get_xlim() # 41~79
ylim = ax.get_ylim() # 49~88

XX = np.linspace(xlim[0], xlim[1], num=100) #41,79
YY = np.linspace(ylim[0], ylim[1], num=100) #49,88
XX,YY = np.meshgrid(XX, YY)
print(XX.shape, YY.shape)

xy = vstack((XX.ravel(), YY.ravel())).T
print(xy.shape)

Z = model.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='black', linewidths=2, alpha=0.5,levels=[-1,0,1],linestyles=['--', '-', '-.'])

plt.contourf(XX,YY,Z,levels=[Z.min(),0,Z.max()], colors=['#FFCCCC','#CCCCFF'],alpha=0.3)

plt.scatter(
    model.support_vectors_[:,0],
    model.support_vectors_[:,1],
    s=100,
    facecolors='none',
    edgecolors='black',
    label="서포트 벡터"
)

plt.title("SVM 비선형(곡선) 분류: 결정경계 및 서포트 벡터 시각화")
plt.legend()
plt.show()

pred = model.predict(x_test)
# 테스터데이터 예측
for x, y in zip(x_test, pred):
    result = '합격' if y else '불합격'
    print(f'시험 점수 {x} -> 예측: {result}')

