import numpy as np
from keras import Sequential
from keras.src.layers import Dense, Input
from tensorflow.python.keras.utils.np_utils import to_categorical

x_data = np.random.random((10, 4))
y_labels = np.random.randint(0,3,size=(10,))

print(f'x_data: {x_data}')
print(f'y_labels: {y_labels}')

# to_categorical 인코딩 (get_dummies 도 사용 가능)

# num_classes : 범주가 3개  [1. 0. 0.]  [0. 1. 0.]  [0. 0. 1.]
y_data = to_categorical(y_labels, num_classes=3)
print(f'y_data: {y_data}')
print()

# Dense 층이 딱 1개 : 입력층(Input Layer) + 출력층(Output Layer) 역할을 동시
model = Sequential([
    Dense(30, activation='relu', input_shape=(4,)),
    Dense(3, activation='softmax')])

# 다중분류 categorical_crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# validation_split=0.3, 30%는 검증용 데이터 분리
model.fit(x_data,y_data, epochs=10, validation_split=0.3, verbose=2)
# 예 데이터 10개
# 학습용 8 / 테스트 2
# 검증용 0.3 => 3

# accuracy: 모델이 얼마나 정답을 맞췄는지 (훈련 데이터 기준)
# val_accuracy: 검증(validation) 데이터 정확도
# 이 값이 0.33 → 0.6 → 0.9 이런 식으로 오르면, 모델이 분류를 점점 잘하고 있다는 뜻

