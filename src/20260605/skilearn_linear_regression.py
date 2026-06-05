import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # 선형 회귀 모델
from sklearn.metrics import mean_absolute_error
## 선형회귀 모델로 예측해보자.

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

## 데이터 전처리

(   train_x, test_x, 
    train_y, test_y    ) = train_test_split(
        perch_length, 
        perch_weight, 
        random_state= 42
    )

# print(train_x.shape, test_x.shape)
# (42,) (14,)

train_x = train_x.reshape(-1,1)
test_x = test_x.reshape(-1,1)

## 모델 준비

LRmodel = LinearRegression()

## 학습

LRmodel.fit(train_x, train_y) # 최적의 w,b가 결정된다.

print('coef: ', LRmodel.coef_, 'intersept = ', LRmodel.intercept_)

# coef(기울기):  [39.01714496] intersept(절편, 편향) =  -709.0186449535477

## 평가

score = LRmodel.score(test_x, test_y)

# print(score)
# 0.8247503123313558.py

## 예측

pred = LRmodel.predict([[30]])
# print(pred)
# [461.49570396]

# print(
#     '(LRmodel.coef_ * 30 + LRmodel.intercept_): ', 
#     (LRmodel.coef_ * 30 + LRmodel.intercept_)
#     )

# (LRmodel.coef_ * 30 + LRmodel.intercept_):  [461.49570396]

## 검증

plt.scatter(perch_length,perch_weight)

def y(x):
    return (LRmodel.coef_*x + LRmodel.intercept_)

def plotDrawer(*x):
    plt.plot(
        list(x),
        list(y(x))
    )

def mark(arg):
    pred_x = arg
    pred_y = LRmodel.predict([[pred_x]])
    plt.scatter(pred_x,pred_y,
                marker='^', 
                c='red')

plotDrawer(15,100)
    
mark(70)


# 직선 방정식의 critical 한 문제. 음수 예측값이 나올 수도 있다.
mark(3)

# 최적의 회귀선이라고 볼 수 없다.

plt.savefig('perchdata.jpeg')