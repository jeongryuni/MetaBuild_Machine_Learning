import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
    'study_time': [2, 3, 4, 5, 6, 8, 10, 12],
    'sleep_time': [9, 8, 8, 7, 6, 6, 5, 5],
    'phone_time': [5, 5, 4, 4, 3, 2, 2, 1],
    'score': [50, 55, 60, 65, 70, 75, 85, 88]
}
# 독립변수 : study_time, sleep_time, phone_time
# 종속변수 : score

df = pd.DataFrame(data, columns=['study_time', 'sleep_time', 'phone_time', 'score'])
x = df.drop('score', axis=1)
y = df['score']

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# 기울기, 절편 (회귀계수)
print(f"기울기 : {model.coef_}")
print(f"절편 : {model.intercept_}")

# 예측점수
print(f"예측점수 :{y_pred}")
df['predicted'] = y_pred
print(df)

# 결정계수(R2)
score = model.score(x, y)
print(f"결정계수(R2) : {score}")

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y, model.predict(x))
print(f"결정계수(R2) : {r2}")

# 테스트 데이터
x_test = pd.DataFrame(
    [[6,7,2],[4,8,4],[10,5,1]],
    columns=['study_time', 'sleep_time', 'phone_time']
)
x_test_pred = model.predict(x_test)
print(f"x_test_pred : {x_test_pred}")

# 평균 제곱오차 MSE (Mean Squared Error) : sum((y-pred)**2) / n
MSE = mean_squared_error(y, y_pred)
print(f"MSE : {MSE}")

# RMSE (Root Mean Squared Error) : sqrt(MSE)
# : MSE가 너무 커질떄 사용
RMSE = np.sqrt(MSE)
print(f"RMSE : {RMSE}")