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

## 데이터 분할

(   train_x, test_x,
    train_y, test_y   ) = \
    train_test_split(
        data_petal, label_petal,
        test_size= 0.2,
        random_state=42,
        shuffle=True
    )

## 모델 준비

cost = 0.7

g = 0.7

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

# train_acc = model_svc.score(train_x.values, train_y.values.ravel())
# test_acc = model_svc.score(test_x.values, test_y.values.ravel())


## 예측

pred = model_svc.predict(test_x)

##성능평가
acc = accuracy_score(test_y, pred)
# print('acc: ',acc)

## train 시각화

# print(iris_dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

# print(iris_dataset['target_names'])
# ['setosa' 'versicolor' 'virginica']

target_names = iris_dataset['target_names']
markers = ['o','^','s']
colors = ['blue', 'green', 'red']

# print(train_y)
for i in set(train_y['label']):
    idx = np.where(train_y['label'] == i)
    # np.where(...) 는 (array([...]),) 형태의 튜플을 반환
    # print(len(idx[0]))
    plt.scatter(
        x = train_x.iloc[idx[0]]['petal_length'],
        y = train_x.iloc[idx[0]]['petal_width'],
        c = colors[int(i)],
        marker= markers[int(i)],
        label = target_names[int(i)] + '(train)',
        s = 80, 
        alpha= 0.3
    )
    # train_x.iloc[idx[0]] 는 그 위치의 행들만 뽑은 DataFrame, 거기서 ['petal_length'] 하면 Series(한 컬럼). 이게 scatter의 x좌표가 됩니다.
    # int(i) 로 감싸는 이유: i가 0.0, 1.0, 2.0 (float)이라 리스트 인덱스(colors[0])로 못 씀. 정수로 변환 필요.

plt.legend(loc='best') # 범례호출 - loc best는 데이터와 겹치지 않는 최적 위치 정해줌.
plt.savefig('figures/iris_svm_01_train_scatter.jpeg')

## test 시각화

for i in set(test_y['label']):
    idx = np.where(test_y['label'] == i)
    plt.scatter(
        x = test_x.iloc[idx[0]]['petal_length'],
        y = test_x.iloc[idx[0]]['petal_width'],
        marker=markers[int(i)],
        label = target_names[int(i)] + '(test)',
        s = 130, # train보다 커서 겹쳐도 테스트가 보이게
        edgecolors='black',
        facecolors = 'none'
    )

plt.legend(loc = 'best')
plt.savefig('figures/iris_svm_02_train_scatter.jpeg')

## 결정경계(등고선) 시각화
# 1) x, y 각 축의 범위 (데이터보다 살짝 넓게 -0.5 ~ +0.5)
x_min = data_petal['petal_length'].min() - 0.5
x_max = data_petal['petal_length'].max() + 0.5
y_min = data_petal['petal_width'].min() - 0.5
y_max = data_petal['petal_width'].max() + 0.5

# 2) 각 축을 100등분
x1s = np.linspace(x_min, x_max, 100)   # x좌표 후보 100개
y1s = np.linspace(y_min, y_max, 100)   # y좌표 후보 100개

# 3) 격자로 펼치기 → 100 x 100 = 10,000개 좌표점
x1, y1 = np.meshgrid(x1s, y1s)

xy1 = np.column_stack([x1.ravel(),y1.ravel()])
Z = model_svc.predict(xy1)
Z = Z.reshape(x1.shape)


# 결정경계 선
plt.contour(x1, y1, Z, levels=[0.0, 1.0, 2.0], colors='red')
# 영역 색칠
plt.contourf(x1, y1, Z, cmap=plt.cm.RdYlBu, alpha=0.3)

plt.xlabel(iris_dataset['feature_names'][2])   # petal length (cm)
plt.ylabel(iris_dataset['feature_names'][3])   # petal width (cm)
plt.title('IRIS : RBF Kernel(C={}, gamma={}), acc: {}'.format(
    cost, g, np.round(acc, 3)))
plt.legend(loc='best')
plt.savefig('figures/iris_svm_03_decision_boundary.jpeg')