from cProfile import label

import sklearn.metrics
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 학습용 데이터
x = np.array([50, 60, 70, 80, 90, 100, 110])
y = np.array([150, 180, 200, 220, 240, 260, 300])

model = LinearRegression()
X = x.reshape(-1, 1)
model.fit(X, y)

# 기울기와 절편 출력
coef = model.coef_[0]
intercept = model.intercept_
print(f"기울기 :{coef}\n절편 : {intercept}")

# y예측값
y_pred = model.predict(X)
print(f"y예측값(y_pred):\n{y_pred}")

# 결정계수
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
print(f"결정계수(r2) : {r2_score(y, y_pred)}")

RSS = np.sum((y - y_pred) ** 2) # 실제값 - 예측값
TSS = np.sum((y - np.mean(y)) ** 2) # 실제값 - 평균값
R2 = 1 - (RSS / TSS)
print(f"R² : {R2:.4f}")


# 그래프 시각화
plt.scatter(x, y, label="실제 데이터")
plt.grid(True, linestyle='--', alpha=0.6)
plt.plot(x, y_pred, color='red', alpha=0.5,  label=f"회귀선: y = {coef:.2f}x + {intercept:.2f}\nR2={r2:.3f}")
plt.xlabel("집 크기")
plt.ylabel("집 가격")
plt.legend()
plt.savefig('ex03.png')
plt.show()


# 테스트 데이터
arr = np.array([55,75,95,105])
arr = arr.reshape(-1, 1)

pred = model.predict(arr)

for i in range(len(arr)):
    print(arr[i],':', pred[i])