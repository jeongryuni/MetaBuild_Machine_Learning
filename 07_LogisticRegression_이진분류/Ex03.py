# 1. 필요한 라이브러리 불러오기
import numpy as np
from keras import Input
from keras.src.layers import Dense, Input
from keras import Sequential
from sklearn.model_selection import train_test_split

# 2. 데이터 불러오기
dataIn = "../00_dataIn/"                    # 데이터 파일이 저장된 폴더 경로
filename = dataIn + "surgeryTest.csv"       # 불러올 CSV 파일 이름 지정
data = np.loadtxt(filename, delimiter=",", skiprows=1)  # CSV 파일을 불러오되, 첫 행(헤더)은 건너뜀
print('data:', data)
print(data.shape)                           # (469, 18) → 469개의 샘플, 18개의 열(피처 + 라벨)

# 3. 입력(X)과 출력(Y) 컬럼 개수 정의
total_col = data.shape[1]       # 전체 열 개수 → 18
y_columns = 1                   # 출력(종속변수, 라벨)은 마지막 열 1개
x_columns = total_col - y_columns  # 입력(독립변수)은 나머지 17개

# 4. 독립변수(X), 종속변수(Y) 분리
# data[:, 0:17] → 0~16번째 컬럼까지 모든 행 (입력데이터)
# data[:, 17]   → 마지막 컬럼만 (정답 라벨)
x = data[:, 0:data.shape[1]-1]   # X: 입력 데이터 (17개 피처)
y = data[:, data.shape[1] - 1]   # Y: 출력 데이터 (정답, 0 또는 1)

#  5. 학습 파라미터 설정
epochs = 30          # 전체 데이터를 몇 번 반복 학습할지 (학습 횟수)
batch_size = 10      # 한 번에 학습할 데이터 묶음 크기

# 6. 학습용/테스트용 데이터 분리
# stratify=y : y의 클래스 비율(0,1)을 train/test에 동일하게 유지
# test_size=0.2 : 전체의 20%를 테스트용으로 사용
# random_state=42 : 랜덤 분할 결과를 고정(재현 가능성)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

# 7. 모델 구성 (Sequential 방식)
# Sequential : 층(Layer)을 순서대로 쌓는 단순한 신경망 구조
model = Sequential([
    Input(shape=(x_columns,)),          # 입력층 → 입력 데이터의 특성 수(17개)
    Dense(30, activation="relu"),       # 은닉층 → 뉴런 30개, ReLU 활성화 함수 사용
    Dense(y_columns, activation="sigmoid"),  # 출력층 → 뉴런 1개, sigmoid(0~1 확률 출력)
])

# 최적의 w와 b를 구하기위함 (손실계산)
# 이진분류 binary_crossentropy
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
fit_hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
print(fit_hist)

# 모델이 각 샘플이 ‘1(양성)’일 확률로 예측한 결과
prediction = model.predict(x_test)
print(f'prediction: {prediction}')

# 0.5 초과 → 1 (양성으로 분류)
# 0.5 이하 → 0 (음성으로 분류)
pred_label = (prediction >0.5).astype(int)
print(f'prediction: {pred_label}')

# Adam: 방향(모멘텀)과 속도조절(RMSprop)을 동시에 사용
# Momentum: 이전 단계의 기울기(gradient) 방향을 “관성”처럼 일정 부분 유지
# RMSprop: 각 파라미터의 기울기 크기를 제곱해 평균내어 학습률을 조정



# [요약]
# - ReLU : 은닉층에서 비선형성 부여 (음수는 0, 양수는 그대로 통과)
# - Sigmoid : 출력층에서 0~1 사이 확률값으로 변환 (이진분류에 사용)
# - Input(shape=(x_columns,)) : 입력 데이터의 피처 수 명시

# ---------------------------------------------
# 참고
# - ReLU (Rectified Linear Unit): 은닉층에서 주로 사용되는 활성화 함수 f(x)=max(0,x)
#   → 0 이하이면 0, 0보다 크면 그대로 출력 (비선형성 부여)
#
# - Sigmoid: 출력값을 0~1 사이 확률로 변환
#   → 예: 0.8이면 "1(양성)"일 확률 80%
#
# - Input(shape=(x_columns,)) :
#   입력 데이터가 몇 개의 특성을 가지는지(특성 수=17)를 명시
# ---------------------------------------------

