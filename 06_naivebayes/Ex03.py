import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sample = '오늘 일정 확인'
okt = Okt()
result = okt.morphs(sample)
# print(f'result : {result}')
# print()

result = okt.morphs('한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다.')
print(f'result : {result}')
# print(' '.join(result))
# print()

print('----------------------------------------------------')
df = pd.read_csv('../00_dataIn/mailList.csv')
print(df.head())


emails = [tuple(row) for row in df.itertuples(index=False)]
print(f'emails:\n{emails}')

# 형태소 분리 후 공백으로 연결하는 함수 생성
def tokenize(text):
    tokens = ' '.join(okt.morphs(text))
    return tokens


# 1️⃣ 데이터 전처리 (형태소 분석 등)/ 형태소 분리 후 ' '공백으로 연결
emails_tokenized = [(tokenize(subject), label) for subject, label in emails]
print(f'emails_tokenized:\n{emails_tokenized}')

# 2️⃣ x, y 분리
x,y = zip(*emails_tokenized)
print(f'x:\n{x}') # subject
print(f'y:\n{y}') # ham or spam
print()

# 3️⃣ 학습/테스트 데이터 분리
x_train_text, x_test_text, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4️⃣ CountVectorizer 학습은 학습데이터로만 (fit)
vectorizer = CountVectorizer()
vectorizer.fit_transform(x_train_text)

# 5️⃣ 변환(transform) 수행
x_train = vectorizer.transform(x_train_text)
x_test = vectorizer.transform(x_test_text)

print(vectorizer.get_feature_names_out())

# 나이브베이즈 : 확률에 기반한 분류 알고리즘
# MultinomialNB : 단어 등장 횟수를 기반
model = MultinomialNB()
model.fit(x_train, y_train)

predict_proba = model.predict_proba(x_test)
print(f'predict_proba : {predict_proba}')
print()
predictions = model.predict(x_test)
print(f'predictions : {predictions}')
print()

new_data = []
f = open('../00_dataIn/checkedMail.csv', encoding='utf-8')
for line in f:
    new_data.append(line.strip())

print(f'new_data : {new_data}')

def tokenize(text):
    tokens = ' '.join(okt.morphs(text))
    return tokens

# 형태소별로 분리

fp = open("../00_dataIn/checkedMail.csv", encoding="utf-8")
new_data = [onemail.strip() for onemail in fp.readlines()]
print(new_data)

fp.close()
final_email_info = []
for new_email in new_data:
    new_email_tokenized = tokenize(new_email)
    new_email_vec = vectorizer.transform([new_email_tokenized])
    pred2 = model.predict(new_email_vec)
    result = f"{new_email} : {pred2[0]}"
    final_email_info.append(result)

print(final_email_info)
