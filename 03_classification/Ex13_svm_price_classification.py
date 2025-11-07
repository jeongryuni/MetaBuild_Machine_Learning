import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#  np.random.normal(평균, 표준편차, 개수)
#  np.random.choice(데이터중 무작위 뽑기)
np.random.seed(42)
data = {
    '면적': np.random.normal(85, 20, 200), # m²,
    '방수': np.random.randint(1, 6, 200),
    '욕실수': np.random.randint(1, 3, 200),
    '연식': np.random.randint(0, 30, 200),
    '지역': np.random.choice(['서울', '부산', '대전', '광주'], 200),
    '용도': np.random.choice(['주거', '상업', '공업'], 200),
    '가격': np.random.normal(4.5, 0.8, 200) * 1e5
}

# 데이터 프레임 생성
df = pd.DataFrame(data)
print(df)

# 결측치 생성
df.loc[np.random.choice(df.index, 10,replace = False),'면적'] = np.nan
df.loc[np.random.choice(df.index, 5,replace = False),'연식'] = np.nan

# 결측치 갯수
print(df.isnull().sum())

# 결측치 제거
df = df.dropna()
print(df.isnull().sum())

# 중복 데이터 행 제거
print(df.shape)
df = df.drop_duplicates()
print(df.shape)

# 가격 중간값 알아내기
price_median = df['가격'].median().round()
print(price_median)

df['고가여부'] = df['가격'] > price_median
df['고가여부'] = df['고가여부'].astype(int)
print(df['고가여부'])

# 지역, 용도 원핫 인코딩
df = pd.get_dummies(df, columns=['지역', '용도'], dtype=int)
print(df)

# 7+1(고가여부)=>8개 칼럼
print(df.columns)

# x = 고가여부가 아닌 컬럼이 독립변수
# y = 고가여부 종속변수
x = df.drop(columns=['가격', '고가여부'])
y = df['고가여부']
print(x.info())
print(y.info())

# 표준화
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 학습데이터/테스트 분리 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 예측 학습
model = SVC(kernel='linear', probability=True)
model.fit(x_train, y_train)

# 테스터 데이터 예측
y_pred = model.predict(x_test)
print(y_pred)

# 실제 데이터
# confusion_matrix()
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 예측확률
proba = model.predict_proba(x_test)
print(proba)

# 임계값(threshold)을 바꿔가며 FPR(위양성률)과 TPR(재현율)을 계산
# y_test와 예측확률을 비교함
#  proba[:, 1] 결과가 0or1로 나오므로 1인 것을 가져옴
fpr, tpr, thresholds = roc_curve(y_test, proba[:, 1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange',lw=3, label=f'ROC curve, AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.1])
plt.ylim([0.0, 1.1])
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

