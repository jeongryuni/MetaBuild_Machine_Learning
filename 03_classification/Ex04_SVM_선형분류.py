# 시험점수 (국어, 영어) => 합격/불합격
import numpy as np
from matplotlib import pyplot as plt
from numpy.ma.extras import vstack
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 시험점수
np.random.seed(111)
# 합격
X_pass = np.random.randn(20, 2) *5 + [70,75]
print('X_pass:\n', X_pass)

# 불합격
X_fail = np.random.randn(20, 2) *5 + [50,55]
print('x_fail:\n', X_fail)

# 두 배열 합치기
x = vstack((X_pass, X_fail))
print('x:\n', x)

y = np.array([1]*20 + [0]*20)
print('y:\n', y)

# 학습/테스트용 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=111)

# SVC모델 학습
model = SVC(kernel='linear')
model.fit(x_train, y_train)

# x테스트의 y예측값
y_pred = model.predict(x_test)
print('y_pred:\n', y_pred)

# 결정경계 그리기
w = model.coef_[0]
b = model.intercept_[0]
x_vals = np.linspace(np.min(x[:,0]-1), np.max(x[:,0]), 100) # x축 설정

# 결정 경계
decision_boundary = -(w[0] * x_vals + b)/ w[1]
# 마진 + 1
margin_positive = -(w[0] * x_vals+ b-1) / w[1]
# 마진 - 1
margin_negative = -(w[0] * x_vals+ b+1) / w[1]

# 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# x[y==0, 0] 불합격자의 국어 점수
# x[y==0, 1] 불합격자의 영어 점수
plt.scatter(x_train[y_train==0, 0], x_train[y_train==0, 1],color='blue', s=100, label='불합격')

# x[y==1, 0] 합격자의 국어 점수
# x[y==1, 1] 합격자의 영어 점수
plt.scatter(x_train[y_train==1, 0], x_train[y_train==1, 1],color='red', s=100, label='합격')
plt.plot(x_vals, decision_boundary,'k', label='결정경계', alpha=0.5, linewidth=5)
plt.plot(x_vals, margin_positive,'k--' ,label='마진+1',  alpha=0.5)
plt.plot(x_vals, margin_negative,'k-.' ,label='마진-1',  alpha=0.5)
plt.scatter(
    model.support_vectors_[:,0],
    model.support_vectors_[:,1],
    s=100,
    facecolors='none',
    edgecolors='black',
    label="서포트 벡터"
)

plt.legend()
plt.title('"SVM 선형분류: 결정경계 및 서포트 벡터 시각화"')
plt.show()

for x, y in zip(x_test, y_pred):
    result = '합격' if y else '불합격'
    print('시험 점수 {} -> 예측: {}'.format(x, result))

# for i, point in enumerate(x_test):
#     print(f"시험점수 {point} => 예측 : {'합격' if y_pred[i]==0 else '불합격'}")

print(f"y_test :{y_test}")
print(f"y_pred: {y_pred}")

# 정확도
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("정확도:", acc)

# new data
x_new = np.array([[80,70],[60,65],[45,50],[72,78]])
x_new_pred = model.predict(x_new)
print("x_new_pred:\n", x_new_pred)

for x, y in zip(x_new, x_new_pred):
    result = '합격' if y else '불합격'
    print(f'시험 점수 {x} -> 예측: {result}')

