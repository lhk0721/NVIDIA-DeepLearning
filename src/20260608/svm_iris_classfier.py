from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm  # svm 모델 추가

iris = load_iris()
#print(iris)

iris_df = pd.DataFrame(np.column_stack( [iris['data'], iris['target']] ),
                       columns=['sepal_len','sepal_wd','petal_len','petal_wd','target'])
print(iris_df)

x_petal_len_wd = iris_df[['petal_len','petal_wd']]
print(x_petal_len_wd.sample(5))
y_target = iris_df[['target']]
print(y_target.sample(5))
#print(y_target.values.ravel()) # ravel() --> 2차원을 1차원으로 변경

cost = 0.3
g=0.7
# train / test 분리
train_x, test_x, train_y, test_y =\
    train_test_split(x_petal_len_wd, y_target, random_state=0)

print(train_x[:5])
print(test_x[:5])

# 모델 준비 ==> SVM 모델로 분류
svm_model = svm.SVC(C = cost, kernel='rbf', gamma=g) # cost = 0.3, g=0.7

# train 데이터 활용해서 모델 학습
svm_model.fit(train_x.values, train_y.values.ravel())

# 성능 평가
print('train acc : ', svm_model.score(train_x.values, train_y.values.ravel()) )
print('test acc : ', svm_model.score(test_x.values, test_y.values.ravel()) )# 테스트 데이터에 대한 성능평가

# 예측
pred = svm_model.predict([[4.7, 1.7]])
print(pred)