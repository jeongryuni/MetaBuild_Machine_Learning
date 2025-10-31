import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

data = {
    '날짜': ['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04', '2023-07-05'],
    '기온(°C)': [30.5, 32.0, 33.5, 31.0, 29.5],
    '강수량(mm)': [0, 5, 0, 10, 0],
    '습도(%)': [55, 60, 65, 70, 50],
    '미세먼지(㎍/㎥)': [35, 45, 50, 60, 30],
    '공휴일여부': [1, 1, 0, 0, 0],
    '여행자 수': [150, 200, 250, 180, 120]
}

df = pd.DataFrame(data)
df['날짜'] = pd.to_datetime(df['날짜'])
df['요일(숫자형)'] = df['날짜'].dt.dayofweek
print(df)

df_corr = df.drop(columns='날짜').corr()
mask = np.triu(np.ones_like(df_corr, dtype=bool))
print(mask) # True False로 나타냄

plt.figure(figsize=(8, 6))
sns.heatmap(df_corr,
            cmap='magma', annot=True, fmt='.2f', mask=mask)
plt.show()