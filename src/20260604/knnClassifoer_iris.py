import numpy as np
import pandas as pd

from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt


## 데이터셋 준비

iris = load_iris()

# print(type(iris))
# <class 'sklearn.utils._bunch.Bunch'>

# print(iris['target_names'])
# ['setosa' 'versicolor' 'virginica']

# print(iris['feature_names'])
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

iris_x = iris['data']
iris_y = iris['target']

iris_xy = np.column_stack((iris_x,iris_y)) # 두개를 튜플로 주면 이어붙인다.

## 데이터프레임 설계

irisDf = pd.DataFrame(
    data=iris_xy,
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width','target'],
)

# print(irisDf)
#      sepal_length  sepal_width  petal_length  petal_width  target
# 0             5.1          3.5           1.4          0.2     0.0
# 1             4.9          3.0           1.4          0.2     0.0
# 2             4.7          3.2           1.3          0.2     0.0
# 3             4.6          3.1           1.5          0.2     0.0
# 4             5.0          3.6           1.4          0.2     0.0
# ..            ...          ...           ...          ...     ...
# 145           6.7          3.0           5.2          2.3     2.0
# 146           6.3          2.5           5.0          1.9     2.0
# 147           6.5          3.0           5.2          2.0     2.0
# 148           6.2          3.4           5.4          2.3     2.0
# 149           5.9          3.0           5.1          1.8     2.0

x = irisDf[['petal_length','petal_width']]
y = irisDf['target']

for i in range(3):
    plt.scatter(
        irisDf.loc[irisDf['target']==i,:]['petal_length'],
        irisDf.loc[irisDf['target']==i,:]['petal_width'],
)

# # plt.scatter(x['petal_length'],x[['petal_width']])
# plt.scatter(x=5, y=2, marker='^') # 2
# plt.scatter(x=1.5, y=0.5, marker='d',c='magenta') # 0
# plt.scatter(x=4, y=1.5, marker='>',c='blue') # 1
# plt.xlabel('petal_length')
# plt.ylabel('petal_width')
# plt.savefig('figures/irisdata.jpeg')

## 모델 준비

knnmodel = KNeighborsClassifier(n_neighbors=1)

knnmodel.fit(x,y)

## 성능평가

# print(knnmodel.score(x,y))
# 0.96 세 품종 중 두 개가 서로 겹치기 때문. 경계 지대에서 나온 오답

## 예측

knn_predict1 = knnmodel.predict([(5,2),(0.5,1.5),(4,1.5)])
# for x in knn_predict1:
#     print(iris['target_names'][int(x)])
# virginica
# setosa
# versicolor

# print(knn_predict1)
# [2. 0. 1.]

knn_predict2 = knnmodel.predict([(5.9,2.3)])

# print(knn_predict2) # virginica

# plt.scatter(x['petal_length'],x[['petal_width']])
# plt.scatter(x=5, y=2, marker='^') # 2
# plt.scatter(x=1.5, y=0.5, marker='d',c='magenta') # 0
# plt.scatter(x=4, y=1.5, marker='>',c='blue') # 1
plt.scatter(x=5.9, y=2.3, marker='^',c='red') # 1
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.savefig('figures/irisdata.jpeg')