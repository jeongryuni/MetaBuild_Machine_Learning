import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 분산 (Variance)
# 정의 : 한 변수의 값들이 평균으로부터 얼마나 퍼져 있는지를 나타내는 지표
# 공식 : Var(X) = Σ(xi - 평균)² / n
# 단위 : 원래 단위의 제곱 (예: cm², 원²)
# 해석 : 값이 클수록 데이터가 평균에서 많이 흩어져 있음
x = [3, 5, 8, 11, 13, 8]
y = [1, 2, 3, 4, 5, 3]

gaesu = len(x)
avg_x = sum(x) / gaesu
avg_y = sum(y) / gaesu

var_x = sum((xi - avg_x)**2 for xi in x) / gaesu   # 모분산
var_y = sum((yi - avg_y)**2 for yi in y) / gaesu   # 모분산
print("분산 var_x:", var_x)
print("분산 var_y:", var_y)
print()

# numpy를 이용한 분산 계산 (ddof=0 → 모집단 기준)
print("np.var(x):", np.var(x, ddof=0))
print("np.var(y):", np.var(y, ddof=0))
print()


# 2. 표준편차 (Standard Deviation)
# 정의 : 분산의 제곱근 (√분산)
# 공식 : Std(X) = √(Σ(xi - 평균)² / n)
# 단위 : 원래 단위 (예: cm, 원)
# 해석 : 평균으로부터 각 값이 떨어진 평균적인 거리
std_x = (sum((xi - avg_x)**2 for xi in x) / gaesu) ** 0.5
std_y = (sum((yi - avg_y)**2 for yi in y) / gaesu) ** 0.5
print("표준편차 std_x:", std_x)
print("표준편차 std_y:", std_y)
print()

# numpy를 이용한 표준편차 계산
print("np.std(x):", np.std(x, ddof=0))
print("np.std(y):", np.std(y, ddof=0))
print()


# 3. 공분산(Covariance)
# np.cov(x,y) 출력 [x분산, 공분산]
# 정의 : 두 변수가 함께 변하는 정도를 나타내는 지표
# 공식 : Cov(X, Y) = Σ[(xi - X평균)(yi - Y평균)] / n
# 단위 : X단위 × Y단위 (예: cm·kg)
# 해석 :
#     ▶ 양수(+) : X가 증가하면 Y도 증가 (정비례 관계)
#     ▶ 음수(-) : X가 증가하면 Y는 감소 (반비례 관계)
#     ▶ 0 근처  : 두 변수는 거의 관계가 없음
cov_xy = sum((xi - avg_x) * (yi - avg_y) for xi, yi in zip(x, y)) / gaesu
print("공분산 cov_xy:", cov_xy)
cov_xy2 = np.cov(x,y, ddof=0)
print("공분산 np.cov(x,y): ", cov_xy2)


# 4. 상관계수 (Correlation Coefficient)
# 정의 : 공분산을 표준편차로 나누어 단위를 없앤 값 (두 변수의 선형 관계 강도)
# 공식 : r = Cov(X, Y) / (σx × σy)
# 범위 : -1 ≤ r ≤ +1
# 해석 :
#     ▶ r = +1 : 완전한 양의 상관관계 (X↑ → Y↑)
#     ▶ r = -1 : 완전한 음의 상관관계 (X↑ → Y↓)
#     ▶ r ≈ 0  : 거의 관계 없음 (독립적이거나 비선형 관계)
# 특징 :
#     • 단위가 없음 (공분산을 표준화했기 때문)
#     • 방향(부호)과 강도(절댓값)을 동시에 나타냄
#     • |r|이 1에 가까울수록 두 변수의 선형관계가 강함
corr = cov_xy / (std_x * std_y)
print("상관계수 corr:", corr)
print("상관계수 corr:", np.corrcoef(x, y))

print('------------------------------------------')
age = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
income = [2500,3000,3500,4000,4500,5000,6000,7000,8000,5000]
print(age, income)

# 분산 : 한 변수의 퍼짐 정도
# 표준편차 : 분산의 제곱근 (실제 단위로 본 퍼짐)
# 공분산 : 두 변수의 함께 변하는 방향
# 상관계수 : 공분산을 표준화해 방향 + 강도까지 표현한 값

# ---------------------평균---------------------
# 평균
age_sum = sum(age)
income_sum = sum(income)

age_mean = age_sum / len(age)
income_mean = income_sum / len(income)
print(f"age_mean: {age_mean}, income_mean : {income_mean}")
# age_mean: 47.5
# income_mean : 4681.818181818182

# ---------------------분산---------------------
# 분산
# Var(X) = Σ(xi - 평균)² / n
diff_sum_age = 0
for a in age:
    diff_sum_age += (a - age_mean)**2 #
var_age = diff_sum_age / len(age)
print(f"var_age :{var_age}")

diff_sum_income = 0
for i in income:
    diff_sum_income += (i - income_mean)**2
var_income = diff_sum_income / len(income)
print(f"var_income: {var_income}")
# var_age :206.25
# var_income: 2785123.9669421487

# ---------------------표준편차---------------------
# 표준편차 : 분산의 제곱근
# Std(X) = √(Σ(xi - 평균)² / n)
age_std = np.sqrt(var_age)
income_std = np.sqrt(var_income)
print(f"age_std : {age_std}, income_std : {income_std}")
# age_std : 14.361406616345072,
# income_std : 1668.8690682441654


# ---------------------공분산---------------------
# 공분산
# Cov(X, Y) = Σ[(xi - X평균)(yi - Y평균)] / n
cov_sum = 0
for a, i in zip(age, income):
    cov_sum += (a - age_mean) * (i - income_mean)
cov = cov_sum / len(age)
print(f"cov_age_income: {cov}")

cov2 = np.cov(age, income, ddof=0)
print("np.cov :", cov2)

# cov_age_income: 20375.000000000004

# ---------------------상관계수---------------------
# 상관계수
# corr =  Cov(X, Y) / (σx × σy)
corr = cov / (age_std * income_std)
print(f"corr: {corr}")
corr2 =np.corrcoef(age, income)
print(f"corr2: {corr2}")
# corr_age_income: 0.8501163590572156

print('--------------------------------------------')

df = pd.DataFrame({'Age': age, 'Income': income})
# corr3 = df['Age'].corr(df['Income'])
corr3 = df.corr()
print(corr3) # 0.8551395146841341

plt.figure(figsize=(8, 6))
sns.heatmap(corr3, annot=True, cmap="YlGnBu", fmt='.2f')
plt.title('Age-Income Correlation Heatmap')
plt.show()
