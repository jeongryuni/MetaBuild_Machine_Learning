import numpy as np
from sklearn.model_selection import train_test_split

x = [[1,2],[3,4],[5,6],[7,8],[9,10]]
y = [0,1,0,1,0]

# random_state : 고정된 테스트 데이터 출력
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42) # 20% 테스트데이터가 됨
print(f"x_train {x_train}")
print(f"x_test {x_test}")
print(f"y_train {y_train}")
print(f"y_test {y_test}")
