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

fish_data = np.column_stack( (length, weight) )
fish_target = np.concatenate( 
    (np.ones(35), np.zeros(14))
)

##
(   train_x,
    test_x, 
    train_y,
    test_y,   ) = train_test_split(
                                    fish_data, 
                                    fish_target, 
                                    stratify=fish_target,
                                    random_state=42)

##

# print(train_x)
# [[  29.7  500. ]
#  [  12.2   12.2]
#  [  33.   700. ]
#  [  11.3    8.7]
#  [  39.5  925. ]
#  [  29.   430. ]
#  ...

mean = np.mean(train_x,0)
# print(mean)
# [ 27.29722222 454.09722222]

std = np.std(train_x,0)
# print(std)
# [  9.98244253 323.29893931]

# 표준점수 = (각 특성데이터 - 평균 / 표준편차)

# print(train_x - mean) # broadcasting
# [[   2.40277778   45.90277778]
#  [ -15.09722222 -441.89722222]
#  [   5.70277778  245.90277778]
#  [ -15.99722222 -445.39722222]
#  [  12.20277778  470.90277778]


trained_scaled = ((train_x - mean)/std)
# print(trained_scaled)
# [[ 0.24070039  0.14198246]
#  [-1.51237757 -1.36683783]
#  [ 0.5712808   0.76060496]
#  [-1.60253587 -1.37766373]
#  [ 1.22242404  1.45655528]

## 학습, 예측 모두 같은 mean,std로 정규화해야 한다. scale 맞추기!!

# newData = ([25, 150]-mean)/std
# print(newData)
# [-0.23012627 -0.94060693]

##

# plt.scatter(trained_scaled[:,0],trained_scaled[:,1])
# plt.scatter(newData[0],newData[1], marker='^', c='red')
# plt.savefig('fishdata_normalized') # 산점도는 같아야 한다. scale이 다른 것 뿐.

## 정규화된 data로학습

knnmodel = KNeighborsClassifier()

knnmodel.fit( trained_scaled, train_y )

## 모델 성능 평가

test_scaled = (test_x -mean)/std

# print(test_scaled)
# [[-1.63258863 -1.37457062]
#  [-1.55244793 -1.37395199]
#  [ 0.24070039 -0.01267317]
#  [-1.55244793 -1.37364268]
#  [-0.07986244 -0.35291555]
#  [-1.4923424  -1.3631261 ]
#  [ 0.67145669  0.71420828]
#  [ 0.67145669  0.3739659 ]
#  [ 1.12224816  1.44108972]
#  [ 0.77163257  0.69874271]
#  [-0.09989762 -0.50757117]
#  [ 0.37092904  0.14198246]
#  [ 1.37268787  1.5338831 ]]

# print('acc: ', knnmodel.score(test_scaled, test_y))
# acc:  1.0

## 완전히 새로운 데이터로 예측
newData = ([25, 150]-mean)/std # 역시 동일한 정규화 

pred = knnmodel.predict([newData]) # 항상 2차원 배열로 받는다. 또는 newData.reshape
# print(pred)
# [1.]