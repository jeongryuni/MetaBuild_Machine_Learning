from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np


x = np.array([
    [0, 1],
    [5, 1],
    [15, 2],
    [25, 5],
    [35, 11],
    [45, 15],
    [55, 34],
    [60, 35]
])
y = np.array([4, 5, 20, 14, 32, 22, 38, 43])

model = LinearRegression()
model.fit(x, y)

# y = wo*xo + w1*x1 + b
print("model_coef_ :", model.coef_)
print("model_coef_ :", model.intercept_)

y_pred = model.predict(x)
print("y_pred :", y_pred)

# 결정계수(R2)
# 1)
RSS = np.sum((y - y_pred) ** 2)
print("RSS :", RSS)
TSS = np.sum((y - np.mean(y)) ** 2)
print("TSS :", TSS)
R2 = 1 - (RSS / TSS)
print("R2 :", R2)

# 2)
from sklearn.metrics import r2_score
R2 = r2_score(y, y_pred)
print("R2 :", R2)

# 3)
R2 = model.score(x, y)
print("R2 :", R2)

# 테스트 데이터
x_test = np.array([[6,5],[8,6],[5,7]])
pred2 = model.predict(x_test)
print("예측점수 :", pred2)

