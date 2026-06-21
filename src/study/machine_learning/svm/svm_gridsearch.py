import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# pd.set_option('display_max_columns', 100)
# pd.set_option('display.width',1000)

## 데이터셋 준비

iris_dataset = load_iris()

iris_df = pd.DataFrame(
    np.column_stack(
        (iris_dataset['data'],
        iris_dataset['target'],)
    ),
    columns=[
        'sepal_length', 'sepal_width',
        'petal_length', 'petal_width',
        'target'
    ]
)

data_petal = iris_df[['petal_length', 'petal_width']]

label_petal = iris_df[['target']].rename(columns = {'target':'label'})

label_petal_flatten = label_petal.values.ravel()

## 교차검증을 이용한 최적의 하이퍼파라미터 찾기
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def svc_param_selection(x,y,nfolds):
    svc_params = [{
        'kernel': ['rbf'],
        'gamma': [0.1,0.3,0.5,0.7,1.0],
        'C': [0.3,0.7,1,1.3,1.5]
    }]

    clf = GridSearchCV(
        SVC(),
        svc_params,
        cv = nfolds
    )
    clf.fit(x,y)

    print('test')
    print(clf.best_params_) # 최적 파라미터
    print(clf.best_score_) # 최적 성능
    print(clf.best_estimator_)  # 최적 모델
    return clf.best_estimator_

svc_model = svc_param_selection(data_petal, label_petal_flatten, 10)
# test
# {'C': 0.3, 'gamma': 0.7, 'kernel': 'rbf'}
# 0.9666666666666666
# SVC(C=0.3, gamma=0.7)
