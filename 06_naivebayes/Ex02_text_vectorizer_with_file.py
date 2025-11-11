from pyexpat import features
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 1️⃣ 텍스트 파일 읽기
words = []
with open('../00_dataIn/text01.txt', 'r', encoding='utf8') as f:
    for line in f:
        words.append(line.strip())   # 줄 끝 개행문자 제거 후 리스트에 추가

# 2️⃣ CountVectorizer로 단어 벡터화
# min_df=2 → 전체 문서 중 최소 2번 이상 등장한 단어만 포함
# stop_words=['세일'] → '세일' 단어는 제외 (불용어 처리)
vectorizer = CountVectorizer(min_df=2, stop_words=['세일'])

# fit_transform() → 단어 사전 생성 + 문서별 단어 등장 횟수 계산
x = vectorizer.fit_transform(words)

# 3️⃣ 단어 사전(vocabulary) 및 결과 확인
print(f'vectorizer.vocabulary_ : \n{vectorizer.vocabulary_}')  # 단어별 인덱스 번호
print(x)                                                       # 희소행렬(sparse matrix)

# 단어 이름(피처명) 출력
features_name = vectorizer.get_feature_names_out(words)
print(features_name)

# 4️⃣ 희소행렬을 배열 형태로 변환 (문서별 단어 등장 횟수)
print(x.toarray())
