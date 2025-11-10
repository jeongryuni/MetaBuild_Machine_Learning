import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

n = 100
np.random.seed(11)

# 정상거래 10시~17시 , 10만원
normal_transactions = np.random.normal([13.5, 10], [3.5, 3.5], (n//2, 2))

# 이상거래 12시~7시, 20만원
abnormal_transactions = np.random.normal([3.5, 20], [3.5, 3.5], (n//2, 2))

# 데이터 합치기
x = np.vstack((normal_transactions, abnormal_transactions))
y = np.array([0]*(n//2) + [1]*(n//2))

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 모델 훈련 및 예측
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 새 데이터
new_transactions = np.array([
    [14, 8],
    [3, 28],
    [9, 18],
])

# 새 데이터 예측
y_pred2 = model.predict(new_transactions)
print(y_pred2)

# 시각화
plt.rc('font', family='malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(x_test[y_test==0, 0], x_test[y_test==0, 1], color = 'b', label = '정상거래')
plt.scatter(x_test[y_test==1, 0], x_test[y_test==1, 1], color = 'r', label = '이상거래')


error = (y_test != y_pred)
plt.scatter(x_test[error,0], x_test[error,1], s=100, facecolors='none', edgecolors='k', label='오분류')
for i in range(len(y_pred2)):
    if y_pred2[i] == 0:
        color = 'g'
        label = f'새 거래 {i + 1} (정상)'
    else:
        color = 'y'
        label = f'새 거래 {i + 1} (이상)'
    plt.scatter(new_transactions[i,0], new_transactions[i,1], color=color, marker='*', label=label, s=100, edgecolors='k')

plt.legend()
plt.grid(True)
plt.xlabel("거래 시각 (시)")
plt.ylabel("거래 금액 (만원)")
plt.title("KNN으로 본 신용카드 이상 거래 탐지")
plt.show()