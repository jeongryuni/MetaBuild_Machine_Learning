from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = sns.load_dataset('titanic')
df.rename(columns={'sex':'gender'},inplace=True)

df = df[['gender', 'survived', 'class']].dropna()

# 엔트로피
survived_count = df['survived'].value_counts()
print()

def entropy_test(probabilities): #[0.5714285714285714, 0.42857142857142855]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

result = []
for (gen, cls), group in df.groupby(['gender', 'class']):
    print('gender:', gen, 'class:', cls)
    print('group:', group)
    counts = Counter(group['survived'])
    print('counts:\n', counts)

    total_count = sum(counts.values())
    ratios = {key: value / total_count for key, value in counts.items()}
    print('ratios:', ratios)

    probabilities = [value for value in ratios.values()]
    print('probabilities:', probabilities)

    entropy = entropy_test(probabilities)
    print('entropy:', entropy)

    result.append({'성별' : gen, '객실등급' : cls, '엔트로피':entropy})

entropy_df = pd.DataFrame(result)
print(entropy_df)

# 그래프 시각화
plt.figure(figsize = (8,5))
sns.barplot(data=entropy_df, x='객실등급', y='엔트로피', hue='성별')
plt.title('Titanic 생존여부의 엔트로피 (성별/객실등급별)')
plt.xlabel('객실등급')
plt.ylabel('엔트로피(bits)')
plt.legend(title='성별')

print()
print('-------------------------------------------------------------------------')

df = sns.load_dataset('titanic')
df = df[['age', 'embarked', 'survived']].dropna()

print(df)


bins = [0, 10, 20, 30, 40, 50, 60, 80]
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']
df['age-group'] = pd.cut(df['age'], bins, labels=labels)

result2 = []
for group, subset in df.groupby('age-group'):
    print(f'group : {group}') # 연령별 그룹명
    print(f'subset : {subset}')

    counts = Counter(subset['survived'])
    print(f'counts:\n{counts}') #

    total_count = sum(counts.values())
    ratios = {key : value/total_count for key, value in counts.items()}
    print(f'ratios:\n{ratios}')

    probabilities = [value for value in ratios.values()]
    print(f'probabilities:\n{probabilities}')
    entropy = entropy_test(probabilities)
    print(f'entropy: {entropy}')

    result2.append({'group' : group, 'entropy' : entropy})

result2_df = pd.DataFrame(result2)
print(result2_df)

