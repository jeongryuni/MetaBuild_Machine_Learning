from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1️⃣ 데이터 준비
docs = [
    "banana apple apple q ??",   # 과일 관련
    "banana grape w q@",          # 과일 관련
    "grape apple banana 123",     # 과일 관련
    "dog cat",                    # 동물 관련
    "dog apple",                  # 애매한 경우
    "cat banana"                  # 애매한 경우
]
labels = [0, 0, 0, 1, 1, 1]  # 0=과일, 1=동물 (가정)

# 2️⃣ 문장 → 벡터로 변환
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(docs)

print("단어 사전:", vectorizer.vocabulary_)
print("x.shape:", x.shape)   # (문장 수, 단어 수)
print(x.toarray())           # 실제 숫자 배열 확인

feature_names = vectorizer.get_feature_names_out()
print(f'feature_names : {feature_names}') # ['123' 'apple' 'banana' 'cat' 'dog' 'grape']

index = list(feature_names).index('apple')
print('index:', index) # apple의 인덱스 1

# 3️⃣ 나이브 베이즈 모델 학습
model = MultinomialNB()
model.fit(x, labels)

# 4️⃣ 새 문장 예측
new_docs = ["apple banana", "dog grape"]
new_x = vectorizer.transform(new_docs)
pred = model.predict(new_x)

print("예측 결과:", pred)  # [0, 1] → 첫 문장은 과일, 두 번째는 동물
print()

# ------------------------------------------------------------
print('-----------------------------------------------------------')
docs = [
    'green red blue',
    'blue red yellow red red' ,
    'red red blue blue'
]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(docs)

print(f'vectorizer.vocabulary_ : {vectorizer.vocabulary_}')

feature_names = vectorizer.get_feature_names_out()
print(f'feature_names : {feature_names}')

print(x.toarray())

