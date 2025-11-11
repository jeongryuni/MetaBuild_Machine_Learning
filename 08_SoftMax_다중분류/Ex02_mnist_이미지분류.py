from keras.src.datasets import mnist
from matplotlib import pyplot as plt

# mnist 모듈에서 제공해주는 이미지 url
# https://observablehq.com/@davidalber/mnist-browser?utm_source=chatgpt.com
# 글씨분류 0 ~ 9 따라서 다중분류 -> softmax
data = mnist.load_data()
print(f'data : {data}')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 60000 이미지개수 /28행 세로(pixel 수) /28열 가로(pixel 수)

print(x_train.shape) # (60000, 28, 28) 3차원 면/행/열
print(y_train.shape) # (60000,)
print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000,)

print(x_train[0], y_train[0]) # 5이미지, 출력 : 5

nrow, ncol = 2, 3
fig, axes = plt.subplots(nrow, ncol, figsize=[10, 6])

# 위치 정하는 공식
# 0행 0열 첫번째이미지, 0행1열 두번쨰 이미지..등등
for idx in range(nrow * ncol):
    ax = axes[idx//ncol, idx%ncol] #idx : 0~5
    ax.imshow(x_train[idx])
    ax.axis('off')
    ax.set_title(f'label: {y_train[idx]}')

plt.show()