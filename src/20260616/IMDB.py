import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB # 다항분포 나이브베이즈
from sklearn.metrics import accuracy_score # 정확도평가
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
np.set_printoptions(threshold=np.inf)

dataSet = pd.read_csv(r'/home/dlgusrb/deeplearning_prg/dataset/IMDB/IMDB Dataset.csv')

# print(dataSet.head())
#                                               review sentiment
# 0  One of the other reviewers has mentioned that ...  positive
# 1  A wonderful little production. <br /><br />The...  positive
# 2  I thought this was a wonderful way to spend ti...  positive
# 3  Basically there's a family where a little boy ...  negative
# 4  Petter Mattei's "Love in the Time of Money" is...  positive

dataSet['sentiment'] = dataSet['sentiment'].map({'positive':1, 'negative':0}).copy()
# print(dataSet.head())
#                                               review  sentiment
# 0  One of the other reviewers has mentioned that ...          1
# 1  A wonderful little production. <br /><br />The...          1
# 2  I thought this was a wonderful way to spend ti...          1
# 3  Basically there's a family where a little boy ...          0
# 4  Petter Mattei's "Love in the Time of Money" is...          1

train_x, test_x, train_y, test_y = train_test_split(
    dataSet['review'], dataSet['sentiment'],
    random_state=41
)

cv = CountVectorizer()
train_cv = cv.fit_transform(train_x)  # 단어사전은 train으로만 학습 (fit_transform)
test_cv = cv.transform(test_x)        # test는 같은 사전으로 변환만 (transform, fit X)
print(train_cv.shape, test_cv.shape)

mnb = MultinomialNB()
mnb.fit(train_cv, train_y)  # MultinomialNB는 희소행렬을 그대로 받음
print('train accuracy:', accuracy_score(train_y, mnb.predict(train_cv)))

pred = mnb.predict(test_cv)
print('test accuracy:', accuracy_score(test_y, pred))

# print(pred)
print(train_cv[:10].toarray())