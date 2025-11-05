# confusion_matrix 혼동행렬
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report

y_test = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1])

'''
[[TN  FP]
 [FN  TP]]
'''
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()
print(f"TN: {TN}\nFP: {FP}\nFN: {FN}\nTP: {TP}")

accuracy = (TP + TN) / (TP + TN + FP + FN)
# accuracy = cm[0][0] + cm[1][1] / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
print("정확도:", accuracy)

precision = TP / (TP + FP)
# precision = cm[1][1] / (cm[1][1] + cm[0][1])
print("정밀도:", precision)

recall = TP / (TP + FN)
# recall = cm[1][1] / (cm[1][1] + cm[1][0])
print("재현율:", recall)

specificity = TN / (TN + FP)
print("특이도:", specificity)

FPR = FN / (FP + TN)
print("FPR:", FPR)

f1_score = 2 * (precision * recall) / (precision + recall)
print("F1-score:", f1_score)

# classification_report(실제 데이터, 예측데이터)
print(classification_report(y_test, y_pred))

