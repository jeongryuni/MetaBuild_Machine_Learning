from pyexpat import features

from sklearn import metrics
from sklearn.metrics import classification_report

# 1. 뉴스 제목 데이터 준비
titles = [
    # 정치
    "대통령 신년 기자회견 열려", "야당 대표, 정부 정책 비판", "정치권, 총선 준비 돌입",
    "국회, 예산안 처리 본격화", "총리, 경제 정책 발표", "정당 간 협상 난항 지속",
    "대선 후보, 공약 발표", "국회의원 선거법 개정 논의", "정부, 대북 정책 강화",

    # 스포츠
    "한국, 월드컵 본선 진출 확정", "류현진, 시즌 첫 승 기록", "손흥민 멀티골로 팀 승리 견인",
    "김민재, 수비수 최초 유럽 리그 우승", "박지성, 전설적인 축구 선수 은퇴", "프로야구, 새 시즌 개막",
    "축구 대표팀, 친선 경기 승리", "테니스 선수, 그랜드슬램 우승", "골프 대회, 상금 기록 경신",

    # 경제
    "주식 시장, 3일 연속 하락", "환율 급등에 수출 기업 비상", "부동산 가격 하락세 지속",
    "코스피, 2500선 회복", "금리 인상에 대출 부담 증가", "수출 호조로 무역 흑자 확대",
    "소비자 물가 상승률 2% 돌파", "은행, 신규 대출 규제 강화", "중소기업 지원 정책 발표"
]

# 각 뉴스 제목의 실제 카테고리 (정답 레이블)
labels = [
    # 정치
    "politics", "politics", "politics",
    "politics", "politics", "politics",
    "politics", "politics", "politics",

    # 스포츠
    "sports", "sports", "sports",
    "sports", "sports", "sports",
    "sports", "sports", "sports",

    # 경제
    "economy", "economy", "economy",
    "economy", "economy", "economy",
    "economy", "economy", "economy"
]

new_titles = [
     "대통령, 외교 정상회담 진행",        # politics
    "코스피, 2600선 돌파하며 상승세",     # economy
    "김민재, 유럽 축구 리그 우승 기록",    # sports
    "환율 변동성 커져 수출업계 긴장",      # economy
    "야당, 새로운 총선 전략 발표",        # politics
    "손흥민, 시즌 20호 골 달성",          # sports
]
# 2. 필요한 라이브러리 불러오기
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 3. 형태소 분석 및 토큰화 함수 정의
okt = Okt()

# 명사(Noun), 동사(Verb), 형용사(Adjective)만 남기고 토큰화
def tokenize(text):
    tokens = okt.pos(text) # 형태소 분석 결과 (단어, 품사) 형태로 반환
    return ' '.join([word for word, pos in tokens if pos in ['Noun', 'Verb', 'Adjective']])

# 모든 뉴스 제목을 형태소 분석 + 토큰화
tokenized_titles = [tokenize(t) for t in titles]
print(tokenized_titles)

# 4. 벡터화 (Bag of Words 방식)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(tokenized_titles)
y = labels

# 5. 데이터 분할 (학습:테스트 = 70:30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# 6. 나이브 베이즈 모델 학습
# MultinomialNB: 단어 등장 횟수를 기반
model = MultinomialNB() # 다항 분포 나이브 베이즈 모델
model.fit(x_train, y_train)

# 7. 테스트 데이터 예측 및 평가
print(f'model.classes_ : {model.classes_}')
predict_proba = model.predict_proba(x_test)
print(f'predict_proba : {predict_proba}')

y_pred = model.predict(x_test)

print(f'x_test : {y_test}')
print(f'y_pred : {y_pred}')

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(f'confusion_matrix : \n{confusion_matrix}')

# 정확도(결정계수)
score = model.score(x_test, y_test)
print(f'score : {score}')

# 정밀도(precision), 재현율(recall), F1-score 등 종합 평가
print(classification_report(y_test, y_pred))

# -----------------------------------------------------------------------
# 8. 새로운 뉴스 제목 예측
new_title_tokenized= [tokenize(new_title) for new_title in new_titles] # 형태소 분석
new_vec = vectorizer.transform(new_title_tokenized) # 벡터화
new_pred = model.predict(new_vec) # 예측
print(f'new_pred : {new_pred}')

# # 새 문장별 예측 결과 보기
for i in range(len(new_title_tokenized)):
    print(f'{new_titles[i]} => {new_pred[i]}')

# 정확도
# 정답을 알고있는 상황에서는 accuracy_score
accuracy_score = metrics.accuracy_score(y_test, y_pred)
print(f'accuracy_score : {accuracy_score}')
