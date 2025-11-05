import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

'''
[컬럼 설명-15가지]
survived: 생존 여부 (0 = 사망, 1 = 생존)
pclass: 티켓 등급(1 = 1등급, 2 = 2등급, 3 = 3등급)
gender: 성별 (male / female)
age: 나이 (float), 일부 결측값 존재
sibsp: 함께 탑승한 형제자매 또는 배우자 수
parch: 함께 탑승한 부모 또는 자녀 수
fare: 탑승 요금 (float)
embarked: 탑승 항구 코드 (C = Cherbourg(셰르부르, 셸부르), Q = Queenstown, S = Southampton)
class: pclass와 동일한 정보지만 범주형(categorical) 타입 (First, Second, Third)
who: 사람 유형 (man, woman, child)
adult_male: 성인 남성 여부 (True/False)
deck: 탑승객이 머무른 데크 (캐빈 위치); 대부분 결측값이 있음
embark_town: 탑승 항구의 도시명 (Southampton, Cherbourg, Queenstown)
alive: 생존 여부를 문자열로 표현 (yes / no)
alone: 혼자 여행 여부 (True = 혼자, False = 동반자 있음)
'''

df = sns.load_dataset('titanic')

# csv파일 저장 : to_csv
df_csv = df.to_csv('../00_dataOut/titanic.csv', index=False)
df = pd.read_csv('../00_dataOut/titanic.csv')

# 컬럼명 변경
df = df.rename(columns={'sex':'gender'})
print(df.info())
# 증복행 개수
print(f"증복행 : {sum(df.duplicated())}")

# 첫 행은 그대로 두고 나머지 삭제
df = df.drop_duplicates()

# deck, embark_town 컬럼 삭제
rdf = df.drop(columns=['deck', 'embark_town'])
print(rdf.columns)
print(rdf.shape)

# age 결측치 제거
rdf = rdf.dropna(subset=['age'])
print(rdf.isnull().sum())

# print(df['embarked'].unique())
# 최빈값 : 가장 많이 나오는 값
rdf_mode = rdf['embarked'].mode()[0]
print(f'rdf_mode: {rdf_mode}')
print()

# 컬럼 embarked 결측치 최빈값으로 채우기
rdf['embarked'] = rdf['embarked'].fillna(rdf['embarked'].mode()[0])
print(rdf.isnull().sum())

# embarked의 데이터 개수 출력
print('embarked value_counts :',rdf['embarked'].value_counts())
print()

ndf = rdf[['survived', 'pclass', 'gender', 'age', 'sibsp', 'parch', 'embarked']]

# 원-핫 인코딩
gender_encode = pd.get_dummies(ndf['gender'], dtype=int)
print(gender_encode)
print()

# 탑승항구(embarked) 원-핫 인코딩 (C, Q, S → town_C, town_Q, town_S)
# prefix='town'으로 컬럼명에 접두사 부여
embarked_encode = pd.get_dummies(ndf['embarked'], prefix='town', dtype=int)
print(embarked_encode)
print()

# 데이터 병합
ndf = pd.concat([ndf, gender_encode, embarked_encode], axis=1)
print(ndf)

ndf = ndf.drop(columns = ['gender', 'embarked'])
print(ndf.columns)

# x 독립변수 : 9개
# y 종속변수 : 1개 [survived]
x = ndf.drop(columns=['survived'])
y = ndf['survived']

# 데이터 표준화
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 테스트, 학습용 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# SVM
# model = SVC(kernel='linear', probability=True)
# probability=True를 작성해야 예측확률을 구할수있음
model = SVC(kernel='linear', probability=True)
model.fit(x_train, y_train)

# 생존예측
y_pred = model.predict(x_test)

# 혼동행렬
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
report = classification_report(y_test, y_pred)
print(report)

# 히트맵 시각화
cm_df = pd.DataFrame(cm, index=['Negative', 'Positive'], columns=['Negative', 'Positive'])
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10,6))
sns.heatmap(cm_df,fmt='d', annot=True, cmap='RdYlGn')
plt.title('혼동행렬 히트맵')
plt.show()

# roc곡선
# 각 클래스(0 또는 1)에 대한 “예측 확률”을 반환
predict_proba = model.predict_proba(x_test)
print(predict_proba) # 2차원 : [0]:사망, [1]:생존 확률
alive_probability = predict_proba[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, alive_probability)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

for i, thresh in enumerate(thresholds):
    if i % 20 == 0 :
        plt.text(fpr[i], tpr[i], f'{thresh:.2f}', fontsize=10)

plt.xlim([0.0, 1.1])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

roc_auc = auc(fpr, tpr)
print(roc_auc)