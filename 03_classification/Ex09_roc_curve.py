import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, auc

y_true = np.random.randint(0, 2, size=100)
y_score = np.concatenate([
    np.random.uniform(0.0, 0.3, 40),   # 음성 그룹 확률
    np.random.uniform(0.7, 1.0, 60)    # 양성 그룹 확률
])

fpr, tpr, thresholds  = roc_curve(y_true, y_score)

auc_score = auc(fpr, tpr)
print('auc_score:', auc_score)

# 그래프 작성
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='darkorange',lw=3, label=f'ROC curve, AUC={auc_score:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('roc curve')
plt.grid(True)
plt.show()