import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {
    'Color':['Red', 'Blue', 'Green', 'Blue', 'Red'],
    'Size': ['S', 'M', 'L', 'M', 'S'],
    'Price': [10, 15, 20, 15, 10],
    'Category' : ['Shoes','Shirts','Pants','Shoes','Shirts']
}

df = pd.DataFrame(data)

# 모든 컬럼표시
pd.set_option('display.max_columns', None)

# 원-핫 인코딩
df_encode = pd.get_dummies(df).astype(int)
print(df_encode)



