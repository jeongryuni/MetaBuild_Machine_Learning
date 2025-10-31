import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

data = {
    '날짜': ['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04', '2023-07-05'],
    '기온(°C)': [30.5, 32.0, 33.5, 31.0, 29.5],
    '여행자 수': [150, 200, 250, 180, 120]
}
df = pd.DataFrame(data)
corr = df[['기온(°C)', '여행자 수']].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
print(mask) # True False로 나타냄

plt.figure(figsize=(8, 6))
sns.heatmap(corr,
            cmap='RdYlGn', annot=True, fmt='.2f', mask=mask)
plt.show()




