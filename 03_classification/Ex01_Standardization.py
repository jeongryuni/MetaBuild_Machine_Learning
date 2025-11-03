import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    'height' : [160,165,170,175,180],
    'weight' : [55,60,65,70,75]
}

df = pd.DataFrame(data)
print(df)

# 표준화
scaler = StandardScaler()
scaler.fit(df)
x_scale = scaler.transform(df) # 1)표준화
x2_scaler = scaler.fit_transform(df) # 2)표준화



print(scaler.mean_)
print(scaler.scale_) #모표준편차

print(df.mean())
print("모표준편차 : ",df.std(ddof=0)) # 모표준편차
print("표본 표준편차 : ",df.std(ddof=1)) # 표본 표준편차

print(f"표준화 평균 : {x_scale.mean()}")
print(f"표준화 표준편차1 : {x_scale.std()}")
print(f"표준화 표준편차2 : {x_scale.std(ddof=0)}")
print(f"표준화 표준편차3 : {x_scale.std(ddof=1)}")