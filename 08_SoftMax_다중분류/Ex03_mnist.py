import numpy as np
from keras import Sequential
from keras.src.datasets import mnist
from keras.src.layers import Dense
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt

data = mnist.load_data()
# print(data)

(x_train, y_train), (x_test, y_test) = data
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 28x28 이미지를 1차원 벡터로 변환하기 위해 열(column) 개수 계산
x_column = x_train.shape[1] * x_train.shape[2]
print('x_column:', x_column)


# -----------------------------
# 데이터 전처리 (Preprocessing)
# -----------------------------
# 1. 평탄화(Flatten)
#    2차원 이미지 데이터를 1차원 벡터로 변환
#    신경망의 Dense Layer는 1차원 입력만 받기 때문
x_train = x_train.reshape(x_train.shape[0], x_column)  # (60000, 784)
x_test = x_test.reshape(x_test.shape[0], x_column)     # (10000, 784)

# 2. 정규화(Normalization)
#    각 픽셀값(0~255)을 0~1 범위로 조정 → 학습 안정성 향상
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 분류해야 할 숫자 클래스 개수 (0~9 → 총 10개)
CLASSES = 10
y_train = to_categorical(y_train, num_classes=CLASSES)
y_test = to_categorical(y_test, num_classes=CLASSES)

# y_train의 첫 번째 값(y_train[0])이 5
# y_train[0] : [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
print(f'y_train[0] : {y_train[0]}')


# Dense : 완전 연결층 (Fully Connected Layer)
# - units=10 : 출력 노드 수 = 분류할 클래스 개수 (숫자 0~9)
# - activation="softmax" : 다중 분류용 활성화 함수 → 각 클래스에 대한 확률을 계산하고, 모든 출력의 합이 1이 되도록 만듦
# - input_shape=(x_column,) : 입력 데이터의 형태 지정 → 784(=28*28)차원 입력 벡터를 받음
model = Sequential()
model.add(Dense(units=512, activation="relu", input_shape=(x_column,))) #출력층
model.add(Dense(units=CLASSES, activation="softmax")) #은닉층

# 모델 구성
# RMSprop: 각 파라미터의 기울기 크기를 제곱해 평균내어 학습률을 조정
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# 학습 데이터를 배치 단위로 나눠 5회 학습하면서 20%는 학습 중 검증용으로 사용
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

score = model.evaluate(x_test, y_test)
print(f'score : {score}') #[손실률, 정확도]

prediction = model.predict(x_test)
print(f'prediction : {prediction}')

# np.argmax()는 배열에서 가장 큰 값의 인덱스(index) 를 반환
predict_class = np.argmax(prediction[0])
print(f'prediction class : {predict_class}')

