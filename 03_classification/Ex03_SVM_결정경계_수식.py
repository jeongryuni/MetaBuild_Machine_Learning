import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 1. 데이터 생성
x_class1= np.array([[2,2],[3,3]])
x_class2= np.array([[-2,-2],[-3,-3]])

x = np.vstack((x_class1,x_class2)) # vstack: 두 배열을 세로로 쌓음 → shape (4,2)
y = np.array([0,0,1,1]) # 클래스 레이블 → 0: 양의 방향, 1: 음의 방향 (x의 데이터가 4개이므로 4개)
print(x) # shape : (4,2)
print(y) # shape : (4,)

#  2. 선형 SVM 모델 생성 및 학습
model = SVC(kernel='linear') # 커널(kernel)을 'linear'로 설정 → 선형 분류기 사용
model.fit(x, y)

# 학습된 파라미터(직선 방정식의 계수와 절편)
# 결정경계식 : w1x1 + w2x2 +b=0
coef = model.coef_[0] # 결정경계의 기울기와 방향(법선 벡터),  가중치 벡터 (w1, w2)
intercept = model.intercept_  #  결정경계의 위치 이동(bias), 절편 b
print(f"model.coef_: {model.coef_}")
print(f"model.intercept_: {model.intercept_}")
print(f"model.support_vectors_:\n{model.support_vectors_}")  # 서포트 벡터 (가장 경계에 있는 점들)
print(f"model.support_vectors_[:,0]:\n{model.support_vectors_[:,0]}")
print(f"model.support_vectors_[:,1]:\n{model.support_vectors_[:,1]}")

# x축의 구간 값을 일정 간격으로 생성 (결정경계와 마진선을 부드럽게 그리기 위함)
# np.min(x[:,0]-1): x의 첫 번째 열(즉, x좌표)의 최소값보다 1 작게 설정 → 그래프 여백 확보
# np.max(x[:,0]): x좌표의 최대값
# 30: 시작~끝 구간을 30등분하여 직선이 매끄럽게 보이도록 설정
x_vals = np.linspace(np.min(x[:,0]-1), np.max(x[:,0]), 100)
print(f"x_vals : {x_vals}")

# 학습된 SVM 모델의 파라미터(결정경계 계수) 추출
# w = [w1, w2]: 가중치(법선 벡터, 직선의 기울기 방향)
# b: 절편(bias), 직선의 위치를 이동시키는 역할
w = model.coef_[0]
b = model.intercept_[0]

# 결정계수 변수들 (plt 생성에 사용)
# SVM의 결정경계(Decision Boundary) 공식
# w1*x1 + w2*x2 + b = 0
# 위 식을 x2(세로축)에 대해 정리하면 → x2 = -(w1*x1 + b) / w2
# 즉, x1에 따른 x2 값을 계산하면 결정경계 직선이 된다.
decision_boundary = -(w[0] * x_vals+ b) / w[1]
print(f"decision_boundary : {decision_boundary}")
print()

# 두개의 마진선 좌표
# SVM의 마진(Margin) 경계선은 결정경계로부터 ±1 떨어진 위치에 존재한다.
# 즉, 결정함수 f(x) = w1*x1 + w2*x2 + b 에 대해
#   f(x) = 0  → 결정경계
#   f(x) = +1 → 한쪽 클래스(양의 마진)
#   f(x) = -1 → 반대쪽 클래스(음의 마진)

# 마진 +1 (결정경계보다 한쪽 방향으로 1만큼 떨어진 선)
# w1*x1 + w2*x2 + b = +1 → x2 = -(w1*x1 + b - 1) / w2
margin_positive = -(w[0] * x_vals+ b-1) / w[1]
print(f"margin_positive : {margin_positive}")

# 마진 -1 (결정경계의 반대쪽으로 1만큼 떨어진 선)
# w1*x1 + w2*x2 + b = -1 → x2 = -(w1*x1 + b + 1) / w2
margin_negative = -(w[0] * x_vals+ b+1) / w[1]
print(f"margin_negative : {margin_negative}") #


# 3. 시각화 설정
# 한글처리/마이너스 처리
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# x축 : x_vals
# y푹 : decision_boundary
# point : 결정경계선 좌표
point = np.vstack((x_vals, decision_boundary)).T
print(f"xy_val: {point}")

# 마진선 좌표 (x, y)
margin_point_pos = np.vstack((x_vals, margin_positive)).T
margin_point_neg = np.vstack((x_vals, margin_negative)).T


# 결정경계와 결정경계/마진 들의 거리 계산
# 결정함수를 구하는 함수값 : model.decision_function()
# 이 값이 0에 가까울수록 실제로 결정경계 위에 있는 점임을 의미함
decision_value_center = model.decision_function(point)
decision_value_pos = model.decision_function(margin_point_pos)
decision_value_neg = model.decision_function(margin_point_neg)

print(f"결정경계 f(x): {decision_value_center}")
print(f"마진 +1 f(x): {decision_value_pos}")
print(f"마진 -1 f(x): {decision_value_neg}\n")

# 시각화
# 클래스별 산점도
# plt.scatter(x[:,0], x[:,1], cmap="coolwarm", c=y, edgecolors='k')
# y = np.array([0, 0, 1, 1])
# 클래스 0 데이터 (y==0)
plt.scatter(x[y==0, 0], x[y==0, 1],
            color='blue', s=100, label='클래스 0', edgecolors='k')
# 클래스 1 데이터 (y==1)
plt.scatter(x[y==1, 0], x[y==1, 1],
            color='red', s=100, label='클래스 1', edgecolors='k')

# 초평면
plt.plot(x_vals, decision_boundary,'k-' ,label="결정경계")
plt.plot(x_vals, margin_positive,'k-.', label="마진+1")
plt.plot(x_vals, margin_negative,'k--', label="마진-1")

plt.legend()
plt.title('SVM 선형분리')
plt.show()

