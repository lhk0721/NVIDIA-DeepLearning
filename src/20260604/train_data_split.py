import numpy as np
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

##
# 더미데이터 ( 캐글Fish Market 데이터 참조 )
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]

bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어데이터 ( 캐글Fish Market 데이터 참조 )
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

## 

# 머신러닝, 딥러닝에서 데이터셋을 학습데이터셋과 테스트데이터셋을 나누어야 한다.
# 데이터 순서도 random하게 shuffle 해야 한다.
# sklearn train_data_split이 그 역할을 한다.
# from sklearn.model_selection import train_model_split

## 

# np.column_stack로 1차원 배열 쌓기

# arr = np.column_stack( ([3,4,5], [8,2,5]) ) # 두 배열을 튜플로 묶어줘야 한다.

# print(arr)
# [[3 8]
#  [4 2]
#  [5 5]]

fish_data = np.column_stack( (length, weight) )
# print(fish_data[:5])
# [[ 25.4 242. ]
#  [ 26.3 290. ]
#  [ 26.5 340. ]
#  [ 29.  363. ]
#  [ 29.  430. ]]

# np.concatenate: 두 배열을 병합해라. pandas와는 다르다.

fish_target = np.concatenate( 
    (np.ones(35), np.zeros(14)) # 튜플로 pack해서 하나로 줘야 한다! concatenate는 하나만 받는다.
)

# print(fish_target)
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0.]

## 데이터셋 분할

# print('fish_data[:5]: ', fish_data[:5])

(   train_x,
    test_x, 
    train_y,
    test_y,   ) = train_test_split(
                                    fish_data, 
                                    fish_target, 
                                    stratify=fish_target,
                                    random_state=42)

# print(len(train_x), len(test_x))
# print(len(train_y), len(test_y))
# 36 13
# 36 13

# print('train_x[:5]: ', train_x[:5])
# fish_data[:5]:  [[ 25.4 242. ]
#  [ 26.3 290. ]
#  [ 26.5 340. ]
#  [ 29.  363. ]
#  [ 29.  430. ]]
# train_x[:5]:  [[ 29.7 500. ]
#  [ 12.2  12.2]
#  [ 33.  700. ]
#  [ 11.3   8.7]
#  [ 39.5 925. ]]

# 전체를 shuffle 한 뒤 train, test 각 비율에 따라 나눔.
# train / test 분할한 데이터셋 준비 완료

## 모델 준비

knnModel = KNeighborsClassifier()

## 학습
knnModel.fit(train_x, train_y)

## 평가
# print(knnModel.score(test_x, test_y))

## 예측
# 임의의 한 데이터를 예측할 때도 2차원 배열 형태로 줘야 한다!! 샘플들의 묶음을 받는 함수기에
print(knnModel.predict([[25, 150]])) # 0: smelt, 1: brim

## 산점도 체크
# plt.scatter()
# print( train_x )
# [0.]
# [[  29.7  500. ]
#  [  12.2   12.2]
#  [  33.   700. ]
#  [  11.3    8.7]
#  [  39.5  925. ]
#  [  29.   430. ]
#  ...

# print( train_x[:,0] )
# [29.7 12.2 33.  11.3 39.5 29.  36.  36.  31.  35.  37.  11.2 34.5 12.
#  29.  33.  30.7 38.5 33.5 14.3 31.5 25.4  9.8 32.  10.5 33.5 10.6 35.
#  32.  35.  13.  30.  32.  15.  30.  41. ]

# print( train_x[:,1] )
# [ 500.    12.2  700.     8.7  925.   430.   714.   850.   475.   720.
#  1000.     9.8  620.     9.8  363.   700.   500.   955.   650.    19.7
#   500.   242.     6.7  600.     7.5  610.     7.   700.   600.   725.
#    12.2  450.   340.    19.9  390.   975. ]

plt.scatter(
    x=train_x[:,0],
    y=train_x[:,1]
)

# plt.scatter(25,150, c='red', marker='^')

# negative 0인 것을 알 수 있다.
# 데이터의 정규화가 되지 않으면 이렇게 잘못된 예측을 한다.

# plt.savefig('figures/fishdata.jpeg')

##
distance, indexes = knnModel.kneighbors([[25,150]])
# arg 기준 거리가 가까운 k개의 데이터 index를 반환.

# print(distance, indexes)
# [[ 92.00086956 130.48375378 130.73859415 138.32150953 138.39320793]] [[21 33 19 30  1]]

# print(train_x) # 현재 train_x처럼 데이터의 범주가 클 굥우 knn 모데르이 문제가 발생할 수 있다.
# 데이터의 범주를 정규화시켜서 적용해야 함.
# plt.scatter( # fancy indexing
#     train_x[indexes,0],
#     train_x[indexes,1],
#     marker='>',
#     c='green'
#     )

# plt.savefig('figures/fishdata.jpeg')

## 표준점수 정규화 필요. standardscaler
# src/20260604/knn_standardScailed.py

#   ┌───────────┬─────────────────┬────────────────┐
#   │           │    5개 이웃     │    KNN 예측    │
#   ├───────────┼─────────────────┼────────────────┤
#   │ 정규화 전 │ 도미 1 + 빙어 4 │ 빙어 ❌ (틀림) │
#   ├───────────┼─────────────────┼────────────────┤
#   │ 정규화 후 │ 도미 5          │ 도미 ✅ (맞음) │
#   └───────────┴─────────────────┴────────────────┘

#   같은 새 생선 (길이 25, 무게 150) 인데, 정규화 하나로 예측이 뒤집혀. 왜 이런지가 정규화의 필요성 그 자체야.

#   왜 정규화 전엔 빙어로 틀렸나

#   유클리드 거리는 모든 축에 똑같은 자(ruler)를 댄다. 수식에서 길이 1 차이와 무게 1 차이가 완전히 동등하게 취급돼:

#   거리² = (길이차)² + (무게차)²

#   그런데 두 특성의 스케일(범위)이 32배 차이나:
#   - 길이 범위: 약 31 (10~41)
#   - 무게 범위: 약 993 (7~1000)

#   그러니 평범한 무게 차이는 수백, 평범한 길이 차이는 수십. 무게 숫자가 압도적으로 커서 거리² 가 사실상 무게로만 결정돼. 위에서 가장 가까운  
#   이웃의 거리를 분해했더니:

#   length 기여 =   0.2   ( 0.0%)
#   weight 기여 = 8464.0  (100.0%)   ← 무게가 거리를 100% 지배

#   즉 정규화 전 KNN은 **사실상 "무게만 보는 1차원 분류기"**야. 길이(25 → 누가 봐도 도미 쪽)는 계산에서 거의 무시돼. 새 생선 무게 150이 숫자상
#   작은 물고기들 쪽에 가깝게 줄세워지면서 이웃 4마리가 빙어로 뽑힌 거지.

#   그림이 주는 착시

#   fishdata.jpeg(원래 스케일)를 보면 y축이 0~1000, x축이 10~41이라 세로축이 엄청 눌려 있어. 네 눈엔 빨간△가 도미 줄기 근처로 보여도,
#   알고리즘이 보는 거리 공간은 저렇게 찌그러진 공간이야. 그래서 초록 이웃들이 아래쪽 작은 물고기로 몰린 거고.

#   fishdata_normalized.png(정규화 후)는 두 축이 똑같이 −1.5~1.7 범위라 공간이 안 찌그러져서, 빨간△가 도미 무리 끝에 제대로 박혀. 이제 길이도 
#   무게도 동등하게 "투표"하니까 이웃 5개가 전부 도미.

#   한 줄 정리

#   ▎ 표준점수 정규화 (x - mean) / std 는 각 특성을 "평균에서 표준편차 몇 칸 떨어졌나" 라는 공통 단위로 바꿔준다. 그래야 거리 기반 모델(KNN,  
#   ▎ K-means, SVM 등)에서 스케일 큰 특성 하나가 거리를 독점하는 걸 막고, 모든 특성이 공평하게 반영된다.

# 오버플로우 방지 차원에서도 정규화는 필수이다. 곱 연산 하다보면 커질 수 있다.

# standardscaler() 를 사용한다.

# dot 연산을 제외한 산술연산은 numpy에서 position별 연산이다. shape를 맞춰줘야 한다.

## scale 정규화 된 특성 데이터 또는 입력 데이터를 활용해서 모델 학습