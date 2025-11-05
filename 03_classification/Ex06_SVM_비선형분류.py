import numpy as np
from matplotlib import pyplot as plt
from numpy.ma.extras import vstack
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

np.random.seed(111)
x_approved = np.random.randn(20, 2) * 5 + [70, 750]
x_denied = np.random.randn(20, 2) * 5 + [50, 600]
x = vstack((x_approved, x_denied))
y = np.array([1]*20 + [0]*20)

# 테스트데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 학습
model = SVC(kernel='rbf')
model.fit(x_train, y_train)

# 결정계수
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8,6))
plt.scatter(x_train[y_train==1,0], x_train[y_train==1,1], color='red', label='승인')
plt.scatter(x_train[y_train==0,0], x_train[y_train==0,1], color='blue', label='거절')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

XX = np.linspace(xlim[0], xlim[1], num=100)
YY = np.linspace(ylim[0], ylim[1], num=100)
XX, YY = np.meshgrid(XX, YY)
xy = vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='black', linewidths=2, alpha=0.5,levels=[-1,0,1],linestyles=['--', '-', '-.'])
plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1],s=100, facecolor='none',edgecolors='black' )
plt.contourf(XX,YY,Z,levels=[Z.min(),0,Z.max()], colors=['#FADADD', '#D6EAF8'],alpha=0.3)
plt.legend()
plt.show()

# 예측
pred = model.predict(x_test)
print(pred)
for x,y in zip(x_test,pred):
    result = '대출 가능' if y else '대출 불가능'
    print(f'급여, 신용점수 {x} -> 예측: {result}')