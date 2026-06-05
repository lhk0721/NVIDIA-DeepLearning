import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train_test 분리
from sklearn.neighbors import KNeighborsRegressor # 특정 값을 예측하는 회귀모델
from sklearn.metrics import mean_absolute_error
##

#농어길이데이터 ( 캐글Fish Market 데이터참조)

perch_length = np.array([
    8.4,13.7,15.0,16.2,17.4,18.0,18.7,19.0,19.6,20.0,21.0,
    21.0,21.0,21.3,22.0,22.0,22.0,22.0,22.0,22.5,22.5,22.7,
    23.0,23.5,24.0,24.0,24.6,25.0,25.6,26.5,27.3,27.5,27.5,
    27.5,28.0,28.7,30.0,32.8,34.5,35.0,36.5,36.0,37.0,37.0,
    39.0,39.0,39.0,40.0,40.0,40.0,40.0,42.0,43.0,43.0,43.5,
    
    44.0])

# 농어무게데이터 (캐글FishMarket 데이터참조)

perch_weight = np.array([
    5.9,32.0,40.0,51.5,70.0,100.0,78.0,80.0,85.0,85.0,110.0,
    115.0,125.0,130.0,120.0,120.0,130.0,135.0,110.0,130.0,
    150.0,145.0,150.0,170.0,225.0,145.0,188.0,180.0,197.0,
    218.0,300.0,260.0,265.0,250.0,250.0,300.0,320.0,514.0,
    556.0,840.0,685.0,700.0,700.0,690.0,900.0,650.0,820.0,
    850.0,900.0,1015.0,820.0,1100.0,1000.0,1100.0,1000.0,
    1000.0])

# 모덱 입력 특성(x)은 농어의 길이. 
# 정답(TargitData, y)는 농어의 무게

# print(len(perch_length, perch_weight))
# 56, 56

## train data와 test data로 분리

(   train_x, 
    test_x, 
    train_y, 
    test_y    ) = train_test_split(
                                    perch_length,
                                    perch_weight,
                                    random_state=42   )

# print(len(train_x), len(test_x))
# 42, 14

## 1차원 shape를 2차원 shape로 바꾸자.

# print(train_x.shape, test_x.shape) 
#  (42,)

train_x = train_x.reshape(-1,1) # -1: data의 갯수만큼 shape를 알아서 지정
test_x = test_x.reshape(-1,1)
# print(train_x.shape, test_x.shape)
# (42, 1) (14, 1)

## knn 회귀모델 준비

knn_reg = KNeighborsRegressor(n_neighbors=3)

## 모델 훈련

knn_reg.fit(train_x, train_y)

## 성능평가

score = knn_reg.score(test_x, test_y)

# print('test score: ',score)
# 0.992809406101064

# print('train score: ',knn_reg.score(train_x,train_y))
# 0.9698823289099254

# train < test ?? 과소적합 문제.
# test든, train이든 90 보다 높고, train이 test보다 약간 높은 것이 좋은 모델이다.

# n=3 준 이후
# test score:  0.9746459963987609
# train score:  0.9804899950518966

## 예측

test_pred = knn_reg.predict(test_x)
# print(test_pred)
# [  60.    79.6  248.   122.   136.   847.   311.4  183.4  847.   113.
#  1010.    60.   248.   248. ]

## 성능평가

mae = mean_absolute_error(test_y, test_pred)
# print(mae)
# 19.157142857142862 ==> 35.42380952380951

## 예측

# 길이가 40인 농어의 길이를 예측하세요

pred = knn_reg.predict([[40]])
# print(pred)
# [921.66666667]

# 길이가 80, 120인 농어의 길이 예측
pred2 = knn_reg.predict([[80],[120]])
# print(pred2)

# [1033.33333333 1033.33333333] 
# knn 문제의 해결 불가능한 문제. 시각화해보자.

pred3 = knn_reg.predict([[50]])
# print(pred3)
# 주변 5개의 평균으로 예측하기에, 길이가 아무리 증가해도 예측 최댓값은 1033.3333이다.

# 해결방법: 주변값 평균이 아닌 데이터 특성을 잘 대변하는 선형회귀, 다항회귀 사용.

##
plt.scatter(train_x,train_y)
plt.scatter(50,1033.33333333,c='red',marker='^')
plt.scatter(80,1033.33333333)
plt.scatter(120,1033.33333333)
plt.savefig('perchdata.jpeg')
