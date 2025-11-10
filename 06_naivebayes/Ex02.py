from pyexpat import features
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

words=[]
with open('../00_dataIn/text01.txt', 'r',encoding='utf8') as f:
    for line in f:
        words.append(line.strip())

# mid_df =2 : 빈도수 2개이상
# stop_words : 해당 단어는 제외
vectorizer = CountVectorizer(min_df=2, stop_words=['세일'])
x = vectorizer.fit_transform(words)

print(f'vectorizer.vocabulary_ : \n{vectorizer.vocabulary_}')
print(x)

features_name = vectorizer.get_feature_names_out(words)
print(features_name)

# 희소행렬
print(x.toarray())

