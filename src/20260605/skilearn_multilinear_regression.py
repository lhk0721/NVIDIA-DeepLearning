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

## 길이 x 에 제곱한 특성을 추가하자.

# print(train_x[:5])
# [[19.6]
#  [22. ]
#  [18.7]
#  [17.4]
#  [36. ]]

train_poly = np.column_stack((train_x**2,train_x))

# print(train_poly[:5])
# [[ 384.16   19.6 ]
#  [ 484.     22.  ]
#  [ 349.69   18.7 ]
#  [ 302.76   17.4 ]
#  [1296.     36.  ]]

test_ploy = np.column_stack((test_x**2, test_x))

## 모델준비

Multi_LR_model = LinearRegression()

## 모델 학습

Multi_LR_model.fit(train_poly,train_y)
# print(Multi_LR_model.coef_, Multi_LR_model.intercept_)
# [  1.01433211 -21.55792498] 116.0502107827827

## 성능평가

score = Multi_LR_model.score(test_ploy,test_y)
# print(score)
# 0.9775935108325122

## 예측
def Predict(arg):
    return Multi_LR_model.predict([[arg**2,arg]])

def y(*x):
    return (1.01*Multi_LR_model.coef_**x -21.6*Multi_LR_model.intercept_ + 116.05)

def plotDrawer(*x):
    plt.plot(
        list(x),
        list(y(x))
    )

def mark(arg):
    pred_x = arg
    pred_y = Multi_LR_model.predict([[pred_x]])
    plt.scatter(pred_x,pred_y,
                marker='^', 
                c='red')



Multi_LR_pred = Predict(30)

# print(Multi_LR_pred)
# [382.21135986]

# print(y(30))
# [ 146.48017406 -530.68753858]

plt.scatter(train_x,train_y)

xpoint = np.arange(15, 50)
ypoint = (1.01*xpoint**2 -21.6*xpoint + 116.05)



plt.plot(xpoint,ypoint)

plt.savefig('perchdata.jpeg')