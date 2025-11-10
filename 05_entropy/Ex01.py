from collections import Counter
import numpy as np

ball_list = ['red', 'blue', 'red', 'blue', 'red', 'blue', 'red']

counter = Counter(ball_list)
print(f'counter : {counter}')
print(f'counter.values() : {counter.values()}')
print(f'sum(counter.values()) : {sum(counter.values())}')
print(f'counter.items() : {counter.items()}')
total_cnt = sum(counter.values())

# 각 색상별 등장 비율(=확률) 계산
# 예: red = 4/7 ≈ 0.5714, blue = 3/7 ≈ 0.4286
ratios = {key : value/total_cnt for key, value in counter.items()}
print(f'ratio : {ratios}')

probabilities = [value for value in ratios.values()]
print(f'probabilities : {probabilities}')


# 엔트로피 공식: H = -Σ (p * log2(p))
# 각 확률에 log2(p)를 취해 곱한 뒤 전체 합의 음수를 취함
# np.array로 자동 변환되어 브로드캐스팅 연산 가능
def entropy_test(probabilities): #[0.5714285714285714, 0.42857142857142855]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 엔트로피 계산
entropy = entropy_test(probabilities)
print(f'entropy : {entropy}')
# 예: 0.9852281360342515 (1에 가까울수록 불확실성이 큼)
# 확률이 균등(즉, 불확실성이 높다) => 엔트로피 값이 커짐

# 예: [0.5, 0.5] → 엔트로피 1.0 (최대 불확실성)
# 예: [1.0, 0.0] → 엔트로피 0 (완전 확실)

# 엔트로피란?
# 엔트로피는 한 사건이 얼마나 예측하기 어려운가를 나타냅니다.
# "불확실성(uncertainty)" 또는 "무질서도(disorder)"의 척도
