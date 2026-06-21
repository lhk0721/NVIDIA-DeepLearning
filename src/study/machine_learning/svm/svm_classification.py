import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # Support Vector Classifier

## 데이터셋 준비

dataset_iris = load_iris()

# print(dataset_iris.keys())
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']

# print(dataset_iris['data'][:5])
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]

# print(dataset_iris['feature_names'])
# # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


iris_df = pd.DataFrame(
    np.column_stack(
        (dataset_iris['data'],
        dataset_iris['target'],)
    ),
    columns=[
        'sepal_length', 'sepal_width',
        'petal_length', 'petal_width',
        'target'
    ]
)
# print(iris_df)
#      sepal_length  sepal_width  petal_length  petal_width  target
# 0             5.1          3.5           1.4          0.2     0.0
# 1             4.9          3.0           1.4          0.2     0.0
# 2             4.7          3.2           1.3          0.2     0.0
# 3             4.6          3.1           1.5          0.2     0.0
# 4             5.0          3.6           1.4          0.2     0.0

data_petal = iris_df[['petal_length', 'petal_width']]
# print (train_x.sample(5))
#      petal_length  petal_width
# 43            1.6          0.6
# 100           6.0          2.5
# 138           4.8          1.8
# 133           5.1          1.5
# 98            3.0          1.1

label_petal = iris_df[['target']].rename(columns = {'target':'label'})
# print(label_petal)
#      label
# 0      0.0
# 1      0.0
# 2      0.0
# 3      0.0
# 4      0.0

# print(label_petal.values)
# [[0.]
#  [0.]
#  [0.]
#  ...

label_petal_flatten = label_petal.values.ravel()
# print(label_petal_flatten)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. ...

## 데이터 분할

(   train_x, test_x,
    train_y, test_y   ) = \
    train_test_split(
        data_petal, label_petal,
        test_size= 0.2,
        random_state=42,
        shuffle=True
    )

# print(train_x[:5])
#     petal_length  petal_width
# 22           1.0          0.2
# 15           1.5          0.4
# 65           4.4          1.4
# 11           1.6          0.2
# 42           1.3          0.2

# print(train_y[:5])
#     label
# 22    0.0
# 15    0.0
# 65    1.0
# 11    0.0
# 42    0.0

## 모델 준비

cost = 0.7
# 규제(regularization) 강도를 조절해. 오분류를 얼마나 허용할지 결정하는 값이야.

g = 0.7
# Radial Basis Function 커널. 데이터를 고차원으로 매핑해서 비선형 경계를 그을 수 있게 해줘. 직선으로 못 나누는 데이터를 곡선으로 나눌 수 있게 되는 거지. 

# C, g 키우면 과적합, 줄이면 과소적합

model_svc = SVC(
    C = cost,
    kernel='rbf',
    gamma=g
)

## 모델 학습

model_svc.fit(
    train_x,
    train_y
)

train_acc = model_svc.score(train_x.values, train_y.values.ravel())
test_acc = model_svc.score(test_x.values, test_y.values.ravel())

# train acc    test acc    진단
# ---------    --------    ----
# 높음          높음         좋은 모델
# 높음          낮음         과적합 (외우기만 함)
# 낮음          낮음         과소적합 (학습 자체가 덜 됨)4

# print( 'train_acc: ', train_acc )
# print( 'test_acc: ', test_acc )
# train_acc:  0.9583333333333334
# test_acc:  1.0

## 예측

pred = model_svc.predict([[4.7, 1.7]])

# print(pred)
# [1.]
