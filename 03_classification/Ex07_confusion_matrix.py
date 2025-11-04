# confusion_matrix 혼동행렬
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

y_test = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])

'''
[[TN  FP]
 [FN  TP]]
'''
cm = confusion_matrix(y_test, y_pred)
print(cm) #

TN, FP, FN, TP = cm.ravel()

# 정확도 :전체 중 맞은 개수
accuracy = (TP + TN) / (TP+TN+FP+FN)
# acc = accuracy_score(y_test, y_pred)

# 정밀도 : 모델이 '양성(1)'이라고 예측한 것 중 실제로 양성인 비율
precision = TP / (TP + FP)
# precision = precision_score(y_test, y_pred)

# 민감도(재현율) : 실제 양성 중에서 모델이 제대로 양성으로 맞춘 비율
sensitivity = TP / (TP + FN)

# 특이도 : 실제 음성 중에서 모델이 제대로 음성으로 맞춘 비율
specificity = TN / (TN + FP)

# F1 Score : Precision과 Recall을 동시에 고려하는 종합 지표
f1_score = 2 * precision * precision / (precision + precision)