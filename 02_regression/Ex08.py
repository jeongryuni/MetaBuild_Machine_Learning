import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../00_dataIn/서울시 부동산 실거래가 정보.csv", encoding="UTF-8")
# print(df.head())

# 결측값 제거
# print(df.isnull().sum())
df = df.dropna()
# print(df.isnull().sum())

x = df[['건물면적(㎡)', '층', '건축년도']]
y = df['물건금액(만원)']

# 학습데이터/테스트데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)

# 기울기(회귀계수:3개)
coef = model.coef_
print(f"기울기 : {coef}")

# 절편
intercept = model.intercept_
print(f"절편 : {intercept}")

# 방정식
print(f"회귀식: y = {coef[0]:.2f}*건물면적 + {coef[1]:.2f}*층 + {coef[2]:.4f}*건축년도 + ({intercept:.2f})")

# 예측값
print(f"예측값 : {pred}")

# 결정계수
# r2_score(y, self.predict(X))
# model.score(x_test, y_test)
R2 = r2_score(y_train, model.predict(x_train)) # 훈련 데이터 R²
print(f"train_R2 : {R2}")

R2 = model.score(x_test, y_test) # 테스트 데이터 R²
print(f"test_R2 : {R2}")

# 가격예측
arr = pd.DataFrame([[84, 5, 2010]],
                   columns=['건물면적(㎡)', '층', '건축년도'])
pred_arr = model.predict(arr)
print(f"가격예측 : {pred_arr[0]:.2f}")

