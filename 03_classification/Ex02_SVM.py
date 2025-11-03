import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

'''
📘 SVM 주요 개념
1) 결정경계(Decision Boundary): 두 클래스를 구분하는 기준선 (decision_function = 0)
2) 마진(Margin): 결정경계와 서포트벡터(Support Vector) 간의 거리
3) 서포트벡터(Support Vector): 마진 경계선 위에 위치한 데이터 포인트들
4) 초평면(Hyperplane): 결정경계를 일반화한 개념 (2D에서는 직선, 3D에서는 평면)
'''


# 1. 데이터 생성
x_class1= np.array([[2,2],[3,3]])
x_class2= np.array([[-2,-2],[-3,-3]])

x = np.vstack((x_class1,x_class2)) # vstack: 두 배열을 세로로 쌓음 → shape (4,2)
y = np.array([0,0,1,1]) # 클래스 레이블 → 0: 양의 방향, 1: 음의 방향
print(x) # shape : (4,2)
print(y) # shape : (4,)


#  2. 선형 SVM 모델 생성 및 학습
model = SVC(kernel='linear') # 커널(kernel)을 'linear'로 설정 → 선형 분류기 사용
model.fit(x, y)

# 학습된 파라미터(직선 방정식의 계수와 절편)
coef = model.coef_[0] # 가중치 벡터 (w1, w2)
intercept = model.intercept_  # 절편 b
print(f"model.coef_: {model.coef_}")
print(f"model.intercept_: {model.intercept_}")
print(f"model.support_vectors_:\n{model.support_vectors_}")  # 서포트 벡터 (가장 경계에 있는 점들)
print(f"model.support_vectors_[:,0]:\n{model.support_vectors_[:,0]}")
print(f"model.support_vectors_[:,1]:\n{model.support_vectors_[:,1]}")


# 3. 시각화 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 클래스별 산점도
plt.scatter(x[:,0], x[:,1], cmap="coolwarm", c=y, edgecolors='k')

# 서포트 벡터 표시 (크게 원으로 강조)
plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=200, edgecolors='k', facecolors='none', label="서포트 벡트")


# 4. 결정경계 계산용 그리드 생성
ax = plt.gca() # 현재 그래프의 축(Axis) 객체 가져오기
xlim = ax.get_xlim() # x축 범위
ylim = ax.get_ylim() # y축 범위

# print(xlim) -> (np.float64(-3.3), np.float64(3.3))
# print(ylim) -> (np.float64(-3.3), np.float64(3.3))

# x, y축을 3등분하여 격자점 생성
xx = np.linspace(xlim[0], xlim[1], 3) # 축길이를 3등분 (방3개 1차원 배열)
yy = np.linspace(ylim[0], ylim[1], 3)
print(f"xx : {xx}")
print(f"yy : {yy}")
# xx : [-3.3  0.   3.3]
# yy : [-3.3  0.   3.3]

# np.meshgrid() : x, y 좌표를 그리드(격자) 형태로 만듦 , 행,열 반복
XX, YY = np.meshgrid(xx, yy)
print(f"XX.shape: {XX.shape}")
print(f"YY.shape: {YY.shape}")
print()

# .ravel() : meshgrid로 만든 2차원 격자 좌표를 1차원으로 펼치는 과정
print(f"XX.ravel(): {XX.ravel()}") #(9,)
print(f"YY.ravel(): {YY.ravel()}") #(9,)
print()

# XX, YY 좌표를 (x, y) 쌍으로 결합
xy = np.vstack([XX.ravel(), YY.ravel()])
print('xy :\n', xy)

# 전치(transpose) → shape: (9,2)
xy = xy.T
print(f"xy.T :\n{xy}") # 결정경계 기준으로 서포트벡트와의 거리

# 노란색 점으로 격자 표시
plt.scatter(xy[:,0], xy[:,1], color="yellow", s=80, edgecolors='k')

# 5. 결정함수 계산 및 등고선 표시
# model.decision_function(xy)
# → 각 점이 결정경계(0)로부터 얼마나 떨어져 있는지를 계산
#   양수면 한쪽 클래스, 음수면 다른 클래스에 속함
print(model.decision_function(xy))  # 9개 거리 값 출력
z = model.decision_function(xy).reshape(XX.shape) # 결정경계선과 좌표와의 거리
print(f"z :\n{z}")

ax.contour(XX, YY, z,
           colors='black',
           linewidths=1,
           alpha=0.5,
           levels=[-1,0,1], # -1 오른쪽 선/ 0 가운데/ 1 왼쪽아래 선
           linestyles=['--', '-', '-.']
           )

plt.legend()
plt.title("선형 SVM 결정경계와 서포트 벡터 시각화")
plt.show()