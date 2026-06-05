import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # 선형 회귀 모델
from sklearn.model_selection import cross_val_score

df = pd.read_csv(r'/home/dlgusrb/deeplearning_prg/dataset/basketball_stat.csv')

## 교차 검증은 성능평가 뿐만 아니라, 최적의 하이퍼 파라미터를 찾는데 쓰일 수 있다.

##
# df.info()
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   Player  100 non-null    object 
#  1   Pos     100 non-null    object 
#  2   3P      100 non-null    float64
#  3   2P      100 non-null    float64
#  4   TRB     100 non-null    float64
#  5   AST     100 non-null    float64
#  6   STL     100 non-null    float64
#  7   BLK     100 non-null    float64
# dtypes: float64(6), object(2)
# memory usage: 6.4+ KB

# print(df.head())
#            Player Pos   3P   2P  TRB  AST  STL  BLK
# 0    Alex Abrines  SG  1.4  0.6  1.3  0.6  0.5  0.1
# 1    Steven Adams   C  0.0  4.7  7.7  1.1  1.1  1.0
# 2   Alexis Ajinca   C  0.0  2.3  4.5  0.3  0.5  0.6
# 3  Chris Andersen   C  0.0  0.8  2.6  0.4  0.4  0.6
# 4     Will Barton  SG  1.5  3.5  4.3  3.4  0.8  0.5

# print(df['Pos'].value_counts())
# Pos
# SG    50
# C     50
# Name: count, dtype: int64

##
# sns.lmplot(
#     x='TRB',
#     y='3P',
#     data=df,
#     fit_reg=False,
#     scatter_kws={'s':150},
#     markers=['o','x'],
#     hue='Pos'
# )
# plt.title('TRB and 3P in 2d plane')

# sns.lmplot(
#     x='BLK',
#     y='3P',
#     data=df,
#     fit_reg=False,
#     scatter_kws={'s':150},
#     markers=['o','x'],
#     hue='Pos'
# )
# plt.title('BLK and 3P in 2d plane')
# plt.savefig('basketball.jpeg')

## 2p, ast, stl 컬럼 삭제
df.drop(columns=['2P','AST','STL'],inplace=True)

# print(df.head(5))
#            Player Pos   3P  TRB  BLK
# 0    Alex Abrines  SG  1.4  1.3  0.1
# 1    Steven Adams   C  0.0  7.7  1.0
# 2   Alexis Ajinca   C  0.0  4.5  0.6
# 3  Chris Andersen   C  0.0  2.6  0.6
# 4     Will Barton  SG  1.5  4.3  0.5

(train, test) = train_test_split(df, test_size=0.2,random_state=45)
# print(train.shape[0]) # 80
# print(test.shape[0]) # 20

max_k_range = train.shape[0] // 2
# print(max_k_range)

k_list = []

for i in range(3,max_k_range,2):
    k_list.append(i)

# print(k_list)
# [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

x_train = train[['3P','BLK','TRB']]
y_train = train[['Pos']]

##

cross_validation_scores = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(
        knn,
        x_train,
        y_train.values.ravel(), # 1차원배열로 펼쳐주기
        cv=10,
        scoring='accuracy'
    )
    cross_validation_scores.append(float(scores.mean()))

# print(cross_validation_scores)
# [0.925, 0.8875, 0.9125, 0.9, 0.8875, 0.8625, 0.8625, 0.8625, 0.8625, 0.8625, 0.8625, 0.875, 0.875, 0.85, 0.8375, 0.8375, 0.8375, 0.825, 0.825]

## 
plt.plot(
    k_list,
    cross_validation_scores
) 

plt.xlabel('number of k')
plt.ylabel('Accracy')
plt.savefig('Basketball_accuracy')

##
(train,test) = train_test_split(
    df,
    test_size=0.2, 
    random_state=49
)

# print(train.shape[0])
# print(test.shape[0])
# 80,20

