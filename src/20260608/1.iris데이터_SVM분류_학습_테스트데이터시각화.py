import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',100)
pd.set_option('display.width',1000)
iris = load_iris()
print(iris)
# sepal : 꽃받침,  petal : 꽃잎
#'data' array : [6.4, 3.1, 5.5, 1.8] sepal_length, sepal_width, petal_length, petal_width
#'target' array : [0,0,0....1,1,1...2,2,2]

# 'target_names': array(['setosa', 'versicolor', 'virginica']
Iris_Data = pd.DataFrame(np.column_stack([iris['data'], iris['target']]),
                         columns=['sepal_len','sepal_wd',
                                  'petal_len','petal_wd','target'])
print(Iris_Data.sample(10))

# #
# petal_len, petal_wd  == > 훈련할 샘플 데이터 추출
X_petal_len_wd = Iris_Data[['petal_len','petal_wd']]
#print(X_petal_len_wd.sample(10))
#
# # 레이블(클래스) 추출
Y_target = Iris_Data[['target']]
#print(Y_target.sample(10))

# # 그리드서치로 최적 c, g 찾아야 함
# 우선 c = 1, g = 0.5 로 고정해서 테스트
c = 1  # cost
g = 0.5 # 감마

X_train, X_test, Y_train, Y_test = train_test_split(X_petal_len_wd, Y_target,
                                                    test_size=0.3, random_state=0)
#print(X_train)
# print(type(Y_train.values.ravel()))
# c, g ==> svm 모델의 분류 결정경계에 영향을 줌
svm_mm = svm.SVC(C = c, kernel='rbf', gamma= g)  # svm 모델 설계

# 0, 1, 2
svm_mm.fit(X_train, Y_train.values.ravel())  # 훈련

Y_pred = svm_mm.predict(X_test) # 예측
print(Y_test.values.ravel()) # 실제값
print(Y_pred) # x_test에 대한 모델 분류 예측값
print('accuracy : ', accuracy_score(Y_test.values.ravel(), Y_pred) )

lnames = iris['target_names']  # 꽃 이름 정보
markers = ['o','^','s']
colors = ['blue','green','red']

#X, Y 좌표(꽃잎 길이, 꽃잎 너비) 학습(train) 데이터 scatter 출력
for i in set(Y_train['target']):  # y_train = [0.0,0.0,..,1.0,1.0,..,2.0,2.0]이므로 중복 없이 0.0 , 1.0, 2.0
    idx = np.where(Y_train['target'] == i)
    print('idx : ', idx)
    print(idx[0])
    # X_train.iloc[idx[0]] : dataframe
    # X_train.iloc[idx[0]]['petal_len'] : dataframe 의 'petal_len'컬럼 선택 : series
    # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 길이(X좌표) 구함
    #print(X_train.iloc[idx[0]]['petal_len'])
    # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 너비(Y좌표) 구함
    #print(X_train.iloc[idx[0]]['petal_wd'])
    # iloc => Fancy indexing 으로 X_train.iloc[idx[0]] , X_train.iloc[idx] 둘다 가능
    plt.scatter(X_train.iloc[idx[0]]['petal_len'], X_train.iloc[idx[0]]['petal_wd'],
                c = colors[int(i)], marker= markers[int(i)],
                label = lnames[int(i)]+'(train)', s=80, alpha=0.3)

# # X, Y 좌표(꽃잎 길이, 꽃잎 너비) 테스트(test) 데이터 scatter 출력
for i in set(Y_test['target']):  # y_test = [0.0,0.0,..,1.0,1.0,..,2.0,2.0]이므로 중복 없이 0.0 , 1.0, 2.0
    idx = np.where(Y_test['target'] == i)
    # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 길이(X좌표) 구함
    #print(X_test.iloc[idx]['petal_len'])
    # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 너비(Y좌표) 구함
    #print(X_test.iloc[idx]['petal_wd'])

    # X, Y 좌표(꽃잎 길이, 꽃잎 너비) 학습 데이터 scatter 출력
    plt.scatter(X_test.iloc[idx]['petal_len'], X_test.iloc[idx]['petal_wd'],
                marker= markers[int(i)], label = lnames[int(i)]+'(test)',
                s=130, edgecolors='black', facecolors = 'none')
    # edgecolors : 경계선 색, facecolors = 'none' : 속을 비움(색칠 x)

plt.legend(loc='best')
plt.savefig('figures/svm_iris.jpeg')