import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, auc

# 실제 정답 (y_test)
# 0: 정상(음성), 1: 암(양성)
y_test = np.array([0, 0, 1, 1])

# 모델이 예측한 "암일 확률값" (연속형)
# 예: 0.8이면 암일 확률이 80%
y_score = np.array([0.1, 0.4, 0.35, 0.8])

# 임계값(threshold) = 0.8 기준
# 0.8 이상이면 암(1), 그 외는 정상(0)으로 분류
y_pred = (y_score >= 0.8).astype(int)
print(f'y_pred:{y_pred}')   # [0 0 0 1]

# 혼동행렬 계산
# 행: 실제값(y_test), 열: 예측값(y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 혼동행렬의 기본 구조:
# [[TN, FP],
#  [FN, TP]]

# 각 값 추출
TN = cm[0][0]   # 실제 0 → 예측 0 (정상 맞게 예측)
FP = cm[0][1]   # 실제 0 → 예측 1 (정상인데 암으로 잘못 예측)
FN = cm[1][0]   # 실제 1 → 예측 0 (암인데 정상으로 잘못 예측)
TP = cm[1][1]   # 실제 1 → 예측 1 (암을 맞게 예측)

# 민감도(TPR, 재현율): 실제 양성 중 양성으로 예측한 비율
TPR = TP / (TP + FN)

# 거짓긍정률(FPR): 실제 음성 중 양성으로 잘못 예측한 비율
FPR = FP / (FP + TN)

print("TPR:", TPR)
print("FPR:", FPR)

print("-----------------------")
# sklearn의 roc_curve 함수를 이용한 TPR, FPR 계산
# threshold를 변화시키며 TPR과 FPR을 자동으로 계산
fpr, tpr, thresholds = roc_curve(y_test, y_score)

print('fpr:', fpr)
print('tpr:', tpr)
print('threshold:', thresholds) # 확률을 내림차순
roc_auc = auc(fpr, tpr)

# AUC : ROC 곡선 아래의 면적값
# → 1에 가까울수록 분류 성능이 좋음
print('roc_auc:', roc_auc) # 0.75

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 각 점에 threshold(기준값) 표시
# - fpr[i], tpr[i] 위치에 해당 threshold[i] 텍스트 출력
for i, thresh in enumerate(thresholds):
    plt.text(fpr[i], tpr[i], f'{thresh:.2f}', fontsize=10)

plt.xlim([0.0, 1.0]) # FPR
plt.ylim([0.0, 1.1]) # TPR
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

