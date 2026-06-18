import pandas as pd
from sklearn.datasets import load_iris # 붓꽃 데이터셋
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # 데이터의 특징이 가우시안 정규분포를 취할 때 분류모델로 사용한다.
# 나이브 베이즈는 조건부확률로 분류를 하는 모델이다.
# 한 사건이 확률했을 때 다른 사건이 발생할 확률이 몇 %인가
from sklearn import metrics # 혼동행렬
from sklearn.metrics import accuracy_score # 정확도평가
import matplotlib.pyplot as plt
import seaborn as sns

## 데이터 특징 확인

dataSet = load_iris()
# print(dataSet)

irisDf = pd.DataFrame(dataSet['data'], columns=dataSet['feature_names'])
# print(irisDf)
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                  5.1               3.5                1.4               0.2
# 1                  4.9               3.0                1.4               0.2
# 2                  4.7               3.2                1.3               0.2
# 3                  4.6               3.1                1.5               0.2
# 4                  5.0               3.6                1.4               0.2
# ..                 ...               ...                ...               ...
# 145                6.7               3.0                5.2               2.3
# 146                6.3               2.5                5.0               1.9
# 147                6.5               3.0                5.2               2.0
# 148                6.2               3.4                5.4               2.3
# 149                5.9               3.0                5.1               1.8

irisDf['target'] = dataSet['target']
# print(irisDf)
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
# 0                  5.1               3.5                1.4               0.2       0
# 1                  4.9               3.0                1.4               0.2       0
# 2                  4.7               3.2                1.3               0.2       0
# 3                  4.6               3.1                1.5               0.2       0
# 4                  5.0               3.6                1.4               0.2       0
# ..                 ...               ...                ...               ...     ...
# 145                6.7               3.0                5.2               2.3       2
# 146                6.3               2.5                5.0               1.9       2
# 147                6.5               3.0                5.2               2.0       2
# 148                6.2               3.4                5.4               2.3       2
# 149                5.9               3.0                5.1               1.8       2

irisDf['target'] = irisDf['target'].map({0:'setosa',1:'versicolor',2:'versinica'})
# print(irisDf)
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)     target
# 0                  5.1               3.5                1.4               0.2     setosa
# 1                  4.9               3.0                1.4               0.2     setosa
# 2                  4.7               3.2                1.3               0.2     setosa
# 3                  4.6               3.1                1.5               0.2     setosa
# 4                  5.0               3.6                1.4               0.2     setosa
# ..                 ...               ...                ...               ...        ...
# 145                6.7               3.0                5.2               2.3  versinica
# 146                6.3               2.5                5.0               1.9  versinica
# 147                6.5               3.0                5.2               2.0  versinica
# 148                6.2               3.4                5.4               2.3  versinica
# 149                5.9               3.0                5.1               1.8  versinica

setosaDf = irisDf.loc[irisDf['target'] == 'setosa'].copy()
versicolorDf = irisDf.loc[irisDf['target'] == 'versicolor'].copy()
versinicaDf = irisDf.loc[irisDf['target'] == 'versinica'].copy()

# print(setosaDf.head())
# print(versicolorDf.head())
# print(versinicaDf.head())
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
# 0                5.1               3.5                1.4               0.2  setosa
# 1                4.9               3.0                1.4               0.2  setosa
# 2                4.7               3.2                1.3               0.2  setosa
# 3                4.6               3.1                1.5               0.2  setosa
# 4                5.0               3.6                1.4               0.2  setosa
#     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)      target
# 50                7.0               3.2                4.7               1.4  versicolor
# 51                6.4               3.2                4.5               1.5  versicolor
# 52                6.9               3.1                4.9               1.5  versicolor
# 53                5.5               2.3                4.0               1.3  versicolor
# 54                6.5               2.8                4.6               1.5  versicolor
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)     target
# 100                6.3               3.3                6.0               2.5  versinica
# 101                5.8               2.7                5.1               1.9  versinica
# 102                7.1               3.0                5.9               2.1  versinica
# 103                6.3               2.9                5.6               1.8  versinica
# 104                6.5               3.0                5.8               2.2  versinica

fig, axis = plt.subplots(1,3,figsize = (15,7))
sns.histplot(data=setosaDf,x='sepal length (cm)',kde=True, ax=axis[0])
sns.histplot(data=versicolorDf,x='sepal length (cm)',kde=True, ax=axis[1])
sns.histplot(data=versinicaDf,x='sepal length (cm)',kde=True, ax=axis[2])
plt.savefig('figures/irisHistplot.jpeg')
# 가우시안 나이즈베이즈 분류에 적합하다는 것을 알 수 있다.