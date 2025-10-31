import numpy as np

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
# 절편은 = y_bar
xy_intercept = y_bar - slope * x_bar
print(xy_intercept)

