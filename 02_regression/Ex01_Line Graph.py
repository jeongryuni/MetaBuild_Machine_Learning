import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

mydata = [(0,1), (1,3), (2,4), (3,5)]
x = [item[0] for item in mydata]
y = [item[1] for item in mydata]

plt.figure(figsize=(8,6))
plt.plot(x, marker='o', linewidth =2, label='직선 그래프')
plt.legend(loc='upper left')
plt.title('Line Graph')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()