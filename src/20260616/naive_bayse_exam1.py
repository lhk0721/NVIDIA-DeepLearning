import pandas as pd
from sklearn.datasets import load_iris # 붓꽃 데이터셋
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # 데이터의 특징이 가우시안 정규분포를 취할 때 분류모델로 사용한다.
# 나이브 베이즈는 조건부확률로 분류를 하는 모델이다.
# 한 사건이 확률했을 때 다른 사건이 발생할 확률이 몇 %인가
from sklearn import metrics # 혼동행렬
from sklearn.metrics import accuracy_score # 정확도평가
import matplotlib.pyplot as plt
import seaborn as sns

##
dataSet = load_iris()
(train_x, test_x, train_y, test_y) =\
    train_test_split(
    dataSet['data'],
    dataSet['target'],
    test_size=0.2,
)

# print((train_x, test_x, train_y, test_y))

## 가우시안 나이브베이즈 모델 준비

GNBmodel = GaussianNB()

## 모델 학습
GNBmodel.fit(train_x,train_y)

## 모델 예측
pred = GNBmodel.predict(test_x)
print('예측: ',pred)
print('실제 정답: ',test_y)
print('test acc: ',GNBmodel.score(test_x, test_y))

## 새로운 데이터 3개만 추가해서 예측해보자.

newData = 