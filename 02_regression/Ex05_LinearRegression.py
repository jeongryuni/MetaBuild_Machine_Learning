from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 학습용 데이터 (70%)
x = np.array([50, 60, 70, 80, 90, 100, 110])
y = np.array([150, 180, 200, 220, 240, 260, 300])
x =x.reshape(-1,1)
# 테스트 데이터 (30%)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

model = LinearRegression()
x_train = x_train.reshape(-1, 1)
model.fit(x_train, y_train)

# 기울기와 절편 출력
coef = model.coef_[0]
intercept = model.intercept_
print(f"기울기 :{coef}\n절편 : {intercept}")

# 테스트 데이터 예측
y_test_pred = model.predict(x_test)

# 결정계수
r2_test = r2_score(y_test, y_test_pred)

print(f"테스트데이터 결정계수(R²) : {r2_test:.4f}")


plt.scatter(x_train, y_train, color='blue', label='훈련 데이터')
plt.scatter(x_test, y_test, color='orange', label='테스트 데이터')
plt.plot(x, model.predict(x), color='red', label=f"회귀선 R2={r2_test:.3f})")

plt.title('선형 회귀 : 훈련/테스트 데이터 비교')
plt.xlabel('집 크기')
plt.ylabel('집 가격')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('ex05.png')
plt.show()
