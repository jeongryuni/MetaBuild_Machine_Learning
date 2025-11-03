import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# x, y 데이터 (두 점)
x = np.array([1, 3])
y = np.array([4, 10])

# 1️⃣ 기울기(slope) 계산
# 공식: (y2 - y1) / (x2 - x1)
# 즉, x가 1 증가할 때 y가 얼마나 증가하는지 계산
a = (y[1] - y[0]) / (x[1] - x[0])   # (10 - 4) / (3 - 1) = 6 / 2 = 3
print("기울기 a:", a)

# 2️⃣ 절편(intercept) 계산
# 공식: y = a*x + b
# 아무 한 점 (x₁, y₁)을 대입해 b 계산 가능
b = y[0] - a * x[0]   # 4 - 3*1 = 1
print("절편 b:", b)

# 3️⃣ 직선 방정식 출력
print(f"직선 방정식: y = {a}x + {b}")

# -------------------------------------
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])


print('기울기')
for i in range(1, len(x)):
    slopes = (y[i] - y[i-1]) / (x[i] - x[i-1])
    print(f"{i}번째 구간 ({x[i - 1]}, {y[i - 1]}) → ({x[i]}, {y[i]}) 의 기울기: {slopes}")

x_bar = np.mean(x)  # 3.0
y_bar = np.mean(y)  # 4.0

# 편차
x_diff = x - x_bar  # [-2, -1, 0, 1, 2]
y_diff = y - y_bar  # [-2,  0, 1, 0, 1]

# 분자
top = np.sum(x_diff * y_diff)
# 분모
bottom = np.sum(x_diff ** 2)

slope = top / bottom
print("분자:", top)
print("분모:", bottom)
print("기울기 slope:", slope)

# 절편
# 절편은 =  y_bar - slope * x_bar
xy_intercept = y_bar - slope * x_bar
print(xy_intercept)

print('--------------------------------------')
# 선형회귀분석
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])


# reshape(-1, 1) 의 뜻
# -1 : 자동 계산(auto). 몇 개의 행이 필요한지 넘파이가 알아서 채움
# 1 : 열(column)의 개수를 1개로 설정 (필수설정)
# x변수 1차원 -> 2차원
X = x.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)  # ← 반드시 학습이 끝난 후

# model.coef_: LinearRegression 모델이 학습을 마치면 자동으로 생성되는 기울기(회귀계수)
# model.intercept_ : 선형회귀식의 “절편(intercept)”, 즉 y축과 만나는 지점(b)
print(f"model.coef_ : {model.coef_}, {model.coef_[0]}") # 기울기
print(f"model.intercept_ : {model.intercept_}") # 절편

# 학습된 회귀식을 이용해 y예측값 계산
# → 실제 y값과 가장 차이가 적은(오차가 최소인) y값들
# y_predict : [2.8 3.4 4.  4.6 5.2]
y_predict = model.predict(X)
print(f"y_predict : {y_predict}")


print("-----------------------결정계수 구하기--------------------------")
# 1) 직접 계산하기
# RSS (Residual Sum of Squares) : 잔차제곱합, 모델의 오차
# → 예측값이 실제데이터에서 얼마나 떨어져있는지
# → (y - y_predict)² 의 합, 값이 작을수록 좋음
RSS = np.sum((y - y_predict) ** 2)

# TSS (Total Sum of Squares) : 전체제곱합(총합산 제곱), 데이터의 전체 변동량
# → 실제 데이터가 평균에서 얼마나 흩어져있는지 정도
#   → (y - y.mean())² 의 합
TSS = np.sum((y - np.mean(y)) ** 2)

# 결정계수 (R2) : 1 - (RSS/TSS)
#   - 1에 가까울수록 모델이 데이터를 잘 설명함
#   - 0이면 예측력이 없음 (y의 평균으로만 예측한 것과 동일)
#   - 완벽한 예측이면 R² = 1
result1_R2 = 1 - (RSS / TSS)
print(f"result1_R2 : {result1_R2}")

# 2) model.score
result2_R2 = model.score(X, y)
print(f"result2_R2 : {result2_R2}")

# 3. sklearn.metrics import r2_score
from sklearn.metrics import r2_score
result3_R2 = r2_score(y, y_predict)
print(f"result3_R2 : {result3_R2}")

print("-------------------------------------------------")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.plot(X, y_predict, 'r',  label='실제 데이터(x,y)')
plt.scatter(X, y, label=f'회귀선 y = {model.coef_[0]} + {model.intercept_}')
plt.grid(True)
plt.title('선형 회귀 : 데이터와 회귀선')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

