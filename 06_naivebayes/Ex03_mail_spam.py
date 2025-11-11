import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1️⃣ 형태소 분석기(Okt) 테스트
sample = '오늘 일정 확인'
okt = Okt()
result = okt.morphs(sample)  # 문장을 형태소 단위로 분리
result = okt.morphs('한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다.')
print(f'result : {result}')
print('----------------------------------------------------')

# 2️⃣ 이메일 데이터 불러오기
df = pd.read_csv('../00_dataIn/mailList.csv')  # 제목과 레이블(ham/spam) 데이터
print(df.head())

# (제목, 라벨) 형태의 튜플 리스트로 변환
emails = [tuple(row) for row in df.itertuples(index=False)]
print(f'emails:\n{emails}')

# 3️⃣ 형태소 분석 및 전처리 함수 정의
def tokenize(text):
    """문장을 형태소 단위로 분리하고 공백으로 연결"""
    tokens = ' '.join(okt.morphs(text))
    return tokens

# 형태소 단위로 변환한 이메일 데이터 생성
emails_tokenized = [(tokenize(subject), label) for subject, label in emails]
print(f'emails_tokenized:\n{emails_tokenized}')

# 4️⃣ 입력(X)과 출력(y) 분리
x, y = zip(*emails_tokenized)
print(f'x:\n{x}')  # 이메일 제목 (문장)
print(f'y:\n{y}')  # 라벨 (ham / spam)
print()

# 5️⃣ 학습용 / 테스트용 데이터 분리 (8:2)
x_train_text, x_test_text, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 6️⃣ CountVectorizer를 이용한 단어 벡터화
vectorizer = CountVectorizer()
vectorizer.fit_transform(x_train_text)      # 학습 데이터로 단어 사전 생성
x_train = vectorizer.transform(x_train_text)  # 학습 데이터 벡터화
x_test = vectorizer.transform(x_test_text)    # 테스트 데이터 벡터화

print(vectorizer.get_feature_names_out())   # 단어 목록 확인

# 7️⃣ 나이브 베이즈 모델 학습 및 예측
# MultinomialNB : 단어 등장 횟수를 이용하는 확률 기반 분류기
model = MultinomialNB()
model.fit(x_train, y_train)

# 테스트 데이터에 대한 확률 및 예측 결과 출력
predict_proba = model.predict_proba(x_test)
print(f'predict_proba : {predict_proba}\n')

predictions = model.predict(x_test)
print(f'predictions : {predictions}\n')

# 8️⃣ 새로운 이메일 데이터 불러오기 (예측용)
new_data = []
f = open('../00_dataIn/checkedMail.csv', encoding='utf-8')
for line in f:
    new_data.append(line.strip())
print(f'new_data : {new_data}')

# 9️⃣ 새 이메일 형태소 분석 후 스팸 여부 예측
fp = open("../00_dataIn/checkedMail.csv", encoding="utf-8")
new_data = [onemail.strip() for onemail in fp.readlines()]
fp.close()
print(new_data)

final_email_info = []
for new_email in new_data:
    new_email_tokenized = tokenize(new_email)                     # 형태소 분석
    new_email_vec = vectorizer.transform([new_email_tokenized])   # 벡터화
    pred2 = model.predict(new_email_vec)                          # 예측 수행
    result = f"{new_email} : {pred2[0]}"                          # 결과 저장
    final_email_info.append(result)

print(final_email_info)  # 최종 예측 결과 출력
