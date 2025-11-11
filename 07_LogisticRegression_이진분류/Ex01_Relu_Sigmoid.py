import numpy as np
import matplotlib.pyplot as plt

# plt 한글 깨질때
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

'''
🔷 활성함수 ⭐⭐⭐
인공신경망은 노드에 입력되는 값을 바로 다음 노드로 전달하지 않고 비선형 함수(활성함수)에 통과시킨 후 전달
어떤 활성함수를 사용하냐에 따라 그 출력값이 달라지므로 적절한 활성함수를 사용하는 것이 중요
대표적인 활성함수 : 시그모이드, 소프트맥스, 렐루
'''

x = np.linspace(-10, 10, 100)
print('x:',x)
print()

sigmoid = 1 / (1+np.exp(-x))
print('sigmoid:',sigmoid)

relu = np.maximum(0, x)
print('relu:',relu)

# 시그모이드
# Sigmoid 함수: 입력값을 0과 1 사이로 매핑합니다. 부드러운 S자 형태를 가집니다. 이진분류
plt.subplot(1, 2, 1)
plt.plot(x,sigmoid, label = '시그모이드', color = 'red', marker = 'o')
plt.legend()

# 렐루
# ReLU 함수: 입력이 0 이상이면 입력값을 그대로 출력하고, 0보다 작으면 0을 출력합니다. 기울기 소실 극복, 이진분류
plt.subplot(1,2,2)
plt.plot(x, relu, label = '렐루', marker = 'o')

plt.legend()
plt.show()