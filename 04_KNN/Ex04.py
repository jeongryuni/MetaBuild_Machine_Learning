import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# np.random.normal(평균, 표준편차, 개수)
# np.random.choice(데이터중 무작위 뽑기)

# x좌표는 평균 1, 표준편차 1
# y좌표는 평균 1, 표준편차 10
# 으로 50개의 (x, y) 난수를 생성
n=100
normal = np.random.normal([1,1], [1.0,10],(n//2,2))
spam = np.random.normal([2.5, 2.5], [1.0,10],(n//2,2))
# 광고 보험료 할인
x = np.vstack([normal, spam])
# 정상 50개 0
# 스팸 50개 1
y = np.array([0]*(n//2) + [1]*(n//2))

# 70/30
# 이웃 5
# 테스트데이터 30개 이웃 5개
# 테스터데이터의 실제 값
# 테서터데이터의 예측 값
# confusion_matrix()해서 실제와 예측값이 다르다면, 점에 테두리 엣지로 검은색주기
# 새로운 이메일은 별로 좌표 그리기
# 정상 : 초록별좌표
# 스팸 : 노란별좌표



# 테스트 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

k_value = 5
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(x_train, y_train)

# 예측 값
y_pred = model.predict(x_test)
print(f"y_pred: {y_pred}")
print(f"y_test: {y_test}")

# FP (False Positive) → 실제 정상인데 스팸으로 잘못 예측 (오탐)
# FN (False Negative) → 실제 스팸인데 정상으로 잘못 예측 (누락)
cm = confusion_matrix(y_test, y_pred).ravel()
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
print(f"TN: {TN}")
print(f"FP: {FP}")
print(f"FN: {FN}")
print(f"TP: {TP}")

# confusion_matrix()해서 실제와 예측값이 다르다면, 점에 테두리 엣지로 검은색주기
# 새로운 이메일은 별로 좌표 그리기
# 정상 : 초록별좌표
# 스팸 : 노란별좌표
new_emails = np.array([
    [1.2, 0.8],
    [3.5, 2.9],
    [2.0, 2.0]
])

pred = model.predict(new_emails)
print(f"new_emails: {pred}")

plt.rc('font', family='malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(x_test[y_test==0, 0], x_test[y_test==0,1], color='blue', marker='o', label='정상')
plt.scatter(x_test[y_test==1, 0], x_test[y_test==1,1], color='red', marker='o', label='비정상')

for i in range(len(pred)):
    if pred[i]==1:
        color = 'yellow'
    else:
        color = 'green'
    plt.scatter(new_emails[i,0], new_emails[i,1], color=color,  marker='*')


plt.legend(loc='best')
plt.show()
