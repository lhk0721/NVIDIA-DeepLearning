import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score # 정확도평가
import matplotlib.pyplot as plt
import seaborn as sns

##

email_list = [
{'email title': 'free game only today', 'spam':True}, 
{'email title': 'cheapest flight deal', 'spam':True}, 
{'email title': 'limited time offer only today only today','spam':True}, 
{'email title': 'today meeting schedule', 'spam':False}, 
{'email title': 'your flight schedule attached', 'spam':False}, 
{'email title': 'your credit card statement', 'spam':False}
]

emaiDf = pd.DataFrame(email_list)
# print(emaiDf)
#                                 email title   spam
# 0                      free game only today   True
# 1                      cheapest flight deal   True
# 2  limited time offer only today only today   True
# 3                    today meeting schedule  False
# 4             your flight schedule attached  False
# 5                your credit card statement  False


# 분류를 위해 라벨을 숫자로 바꿔주자.
emaiDf['spam'] = emaiDf['spam'].map({True:1, False:0})
# print(emaiDf)
#                                 email title  spam
# 0                      free game only today     1
# 1                      cheapest flight deal     1
# 2  limited time offer only today only today     1
# 3                    today meeting schedule     0
# 4             your flight schedule attached     0
# 5                your credit card statement     0

train_x = emaiDf['email title']
train_y = emaiDf['spam']

##
cv = CountVectorizer(binary=True)
train_x_cv = cv.fit_transform(train_x)
# print(train_x_cv)
# <Compressed Sparse Row sparse matrix of dtype 'int64'
#         with 23 stored elements and shape (6, 17)>
#   Coords        Values
#   (0, 6)        1
#   (0, 7)        1
#   (0, 11)       1
#   (0, 15)       1
#   (1, 2)        1
#   (1, 5)        1
#   (1, 4)        1
#   (2, 11)       1
#   (2, 15)       1
#   (2, 8)        1
#   (2, 14)       1
#   (2, 10)       1
#   (3, 15)       1
#   (3, 9)        1
#   (3, 12)       1
#   (4, 5)        1
#   (4, 12)       1
#   (4, 16)       1
#   (4, 0)        1
#   (5, 16)       1
#   (5, 3)        1
#   (5, 1)        1
#   (5, 13)       1

train_encoded = train_x_cv.toarray()
# print(train_encoded)
# [[0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 0]
#  [0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 1 0 1 1 0 0 1 1 0]
#  [0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0]
#  [1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1]
#  [0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1]]

# print(cv.get_feature_names_out())
# ['attached' 'card' 'cheapest' 'credit' 'deal' 'flight' 'free' 'game'
#  'limited' 'meeting' 'offer' 'only' 'schedule' 'statement' 'time' 'today'
#  'your']

## 모델 준비 
# 베르누이 나이즈베이즈 분류는 특성데이터가 0,1로 이루어져있어야 한다.

bnb = BernoulliNB()

## 모델 학습

train_y = train_y.astype('int')
bnb.fit(train_encoded,train_y)

# print('acc: ', bnb.score(train_encoded, train_y))
# acc:  1.0

##

temp_mail_cv = cv.transform(['last discount event of today', 'the payment document is attached and sent', 'company collaboration event free offer'])
# print(temp_mail_cv.toarray())
# [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
#  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0]]

predict = bnb.predict(temp_mail_cv)
# print(predict)
# [1 0 1]