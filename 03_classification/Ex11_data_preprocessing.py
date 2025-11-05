import numpy as np
import pandas as pd

data = {
    "Name" : ['kim', 'park', 'choi', 'choi','kim', np.nan, 'park'],
    "Age" : [30, 30, 25, np.nan, 30, 20, 29],
    "Address" : ['Jeju', np.nan, "Jeju", "Seoul", "Jeju", "Seoul", "Busan"],
}

df = pd.DataFrame(data)

print(f"증복행 : {sum(df.duplicated())}")

print(f"df:{df}")
# 중복제거 (전체 컬럼)
df_duplicated = df.drop_duplicates()
print(f"df_duplicated:{df_duplicated}")

# 중복제거 (지정 컬럼)
df_duplicated_name = df.drop_duplicates(subset=['Name'])
print(f"df_duplicated_name:{df_duplicated_name}")

# 중복제거 (마지막 유지)
df_duplicated_last = df.drop_duplicates(subset=['Name'], keep="last")
print(f"df_duplicated_last:{df_duplicated_last}")

# 결측치 제거
df_na = df.dropna()
print(f"df_na:{df_na}")

# 지정한 컬럼 결측값 제거
df_na = df.dropna(subset=['Name'])
print(f"df_na:{df_na}")

# 멀티 컬럼 결측값 제거
df_na = df.dropna(subset=['Name','Age'])
print(f"df_na:{df_na}")

dupe_mask = df.duplicated(keep=False)
print('dupe mask:\n', dupe_mask)
print()

