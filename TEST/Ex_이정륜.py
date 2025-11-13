import pandas as pd
import numpy as np
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  StandardScaler
from sklearn.svm import SVC

n_samples = 200
data = pd.DataFrame({
    '칼로리': np.random.randint(1500, 3000, n_samples),
    '단백질': np.random.randint(40, 150, n_samples),
    '지방': np.random.randint(30, 100, n_samples),
    '탄수화물': np.random.randint(150, 400, n_samples),
    '운동빈도': np.random.randint(0, 7, n_samples),  # 주당 횟수
    '수면시간': np.random.randint(4, 10, n_samples),
    '건강지표': np.random.randint(50, 100, n_samples),  # 종합 건강 점수
    '선호식품': np.random.choice(['과일', '채소', '고기', '음료', '간식'], n_samples),
    '질병유무': np.random.choice([0,1], n_samples)  # 0: 없음, 1: 있음
})

# 1. 일부 결측치 삽입
random.seed(111)
rand_vals = np.random.rand(n_samples)

nan_idx = np.where(rand_vals < 0.1)[0]
print(len(nan_idx))
data.loc[nan_idx, '수면시간'] = np.nan
print(f'결측치 처리 전 : {data.isnull().sum()}')

# 2. 결측치 처리
# 수면시간의 nan자리에 수면시간의 평균 넣기
data['수면시간'] = data['수면시간'].fillna(data['수면시간'].mean())
print(f'결측치 처리 후 : {data.isnull().sum()}')


# 3. 선호 식품 원-핫 인코딩해서 숫자로 변경
food_df = pd.get_dummies(data['선호식품'], prefix='선호식품').astype(float)
df = pd.concat([data, food_df], axis=1)
df = df.drop(columns='선호식품')

# ---------------------------------------------------------------------------
# 4. 표준화
# 질병유무 제외하고 표준화
scaler = StandardScaler()
not_scale = df['질병유무']
df_drop_not_scale = df.drop(columns='질병유무')

df_scaled = scaler.fit_transform(df_drop_not_scale)
df_scaled = pd.DataFrame(df_scaled, columns=df_drop_not_scale.columns)

df = pd.concat([df_scaled, not_scale], axis=1)

# 5. 표준화한 데이터 분리 : 학습용 데이터, 테스트 데이터(30%)
# 독립변수 : 질병유무 빼고 모두
# 종속변수 : 질병유무(표준화 안되어 있음)
x = df.drop(columns='질병유무')
y = df['질병유무']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 6. KNN 분류
# 테스트 데이터 이웃 5개 보고 질병유무 예측
# 테스트 데이터의 결과와 실제 데이터 비교하는 confusion matrix 출력
k = 5
model_knn = KNeighborsClassifier(n_neighbors=k)
model_knn.fit(x_train, y_train)
y_pred_knn = model_knn.predict(x_test)
print(f'질병예측 knn : \n{y_pred_knn}')

cm_knn = confusion_matrix(y_test, y_pred_knn)
print(f'cm_knn : \n{cm_knn}')

# 7. SVM 분류
# confusion matrix  출력
model_svm = SVC()
model_svm.fit(x_train, y_train)
y_pred_svm = model_svm.predict(x_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(f'cm_svm : \n{cm_svm}')

#
# ---------------------------------------------------------------------------
# 8. MultinomialNB + CountVectorizer

texts = ['식후 항상 과일',  '채소를 즐겨 먹음',  '고기 위주 식사',  '음료 많이 마심',  '간식 즐김']
texts = np.random.choice(texts, n_samples)
y = np.random.choice([0,1], 200)

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(texts)
print(vectorizer.vocabulary_)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model_nb = MultinomialNB()
model_nb.fit(x_train, y_train)
y_pred_nb = model_nb.predict(x_test)
print(f'y_pred_nb : {y_pred_nb}')
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(f'cm_nb : {cm_nb}')

# ---------------------------------------------------------------------------
# 9. DNN 분류 (Softmax)
#
# 표준화한 학습 데이터로
# Sequential 생성
# 입력은 학습데이터의 모든 칼럼수,
# 은닉층을 통해 출력되는 갯수(뉴런수): 64units 출력 뉴런 수, 활성화함수:relu
# 출력층을 통해 출력되는 갯수(뉴런수): 2units 출력 뉴런 수, 2가지의 확률값이 출력된다는 뜻, 활성화함수:softmax
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# epoch=20,
# batch_size=10
#
# 2가지 확률값중 더 큰값을 예측값으로 출력하기
# confusion matrix 출력
#



# ---------------------------------------------------------------------------
# 3개의 클러스터 생성
# data뒤에 Cluster 칼럼 추가
#
#  칼로리  단백질  지방  탄수화물  운동빈도      수면시간  건강지표  질병유무  선호식품_간식  선호식품_고기  선호식품_과일  선호식품_음료  선호식품_채소  Cluster
# 0    2626  144  52   315     1  6.714286    55     1      1.0      0.0    0.0      0.0      0.0        0
# 1    2959   91  95   232     3  8.000000    57     1      0.0      0.0    0.0      1.0      0.0        2
#
# 그래프 그리기
# ---------------------------------------------------------------------------
# 각 클러스터 별 평균 구하기
#   칼로리    단백질     지방    탄수화물  운동빈도  수면시간   건강지표  질병유무  선호식품_간식   선호식품_고기  선호식품_과일  선호식품_음료  선호식품_채소
# Cluster
# 0        2346.79  92.81  62.79  279.47  3.42  6.69  73.60  0.43     0.51   0.49      0.0     0.00     0.00
# 1        2283.95  87.95  64.14  283.21  2.40  6.87  76.51  0.49     0.00   0.00      1.0     0.00     0.00
# 2        2292.67  95.88  65.99  276.30  2.87  6.65  74.38  0.49     0.00   0.00      0.0     0.39     0.61
#
#
# ---------------------------------------------------------------------------
#
#
#
