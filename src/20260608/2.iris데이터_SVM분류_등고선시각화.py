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

Iris_Data = pd.DataFrame(np.column_stack([iris['data'], iris['target']]),
                         columns=['sepal_len','sepal_wd',
                                  'petal_len','petal_wd','target'])
#print(Iris_Data.sample(10))

# petal_len, petal_wd  == > X 데이터 추출
X_petal_len_wd = Iris_Data[['petal_len','petal_wd']]
print(X_petal_len_wd.sample(10))

# 레이블(클래스) 추출
Y_target = Iris_Data[['target']]
#print(Y_target.sample(10))

# 그리드서치로 최적 c, g 찾아야 함
# 우선 c = 1, g = 0.5 로 고정해서 테스트
c = 1
g = 0.5
X_train, X_test, Y_train, Y_test = train_test_split(X_petal_len_wd, Y_target,
                                                    test_size=0.3, random_state=0)
#print(X_train)
#print(Y_train)
svm_mm = svm.SVC(C = c, kernel='rbf', gamma= g)

svm_mm.fit(X_train, Y_train.values.ravel())  # 훈련

print(X_test)
Y_pred = svm_mm.predict(X_test) # 예측
print(Y_test.values.ravel())
print(Y_pred)
accuracy  = accuracy_score(Y_test.values.ravel(), Y_pred)
print('accuracy : ',  np.round(accuracy,3))

lnames = iris['target_names']  # 꽃 이름 정보
markers = ['o','^','s']
colors = ['red','yellow','blue']

# # X, Y 좌표(꽃잎 길이, 꽃잎 너비) 학습(train) 데이터 scatter 출력
# for i in set(Y_train['target']):  # y_train = [0.0,0.0,..,1.0,1.0,..,2.0,2.0]이므로 중복 없이 0.0 , 1.0, 2.0
#     idx = np.where(Y_train['target'] == i)
#     # X_train.iloc[idx] : dataframe
#     # X_train.iloc[idx]['petal_len'] : dataframe 의 'petal_len'컬럼 선택 : series
#     # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 길이(X좌표) 구함
#     #print(X_train.iloc[idx]['petal_len'])
#     # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 너비(Y좌표) 구함
#     #print(X_train.iloc[idx]['petal_wd'])

#     plt.scatter(X_train.iloc[idx]['petal_len'], X_train.iloc[idx]['petal_wd'],
#                 c = colors[int(i)], marker= markers[int(i)], edgecolors='black',
#                 label = lnames[int(i)]+'(train)', s=80)

# X, Y 좌표(꽃잎 길이, 꽃잎 너비) 테스트(test) 데이터 scatter 출력
for i in set(Y_test['target']):  # y_test = [0.0,0.0,..,1.0,1.0,..,2.0,2.0]이므로 중복 없이 0.0 , 1.0, 2.0
    idx = np.where(Y_test['target'] == i)

    # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 길이(X좌표) 구함
    #print(X_test.iloc[idx]['petal_len'])
    # 학습데이터 타깃과 일치한 인덱스의 학습데이터 꽃잎 너비(Y좌표) 구함
    #print(X_test.iloc[idx]['petal_wd'])

    # X, Y 좌표(꽃잎 길이, 꽃잎 너비) 학습 데이터 scatter 출력
    plt.scatter(X_test.iloc[idx]['petal_len'], X_test.iloc[idx]['petal_wd'],
                marker= markers[int(i)], label = lnames[int(i)]+'(test)',
                s=130, edgecolors='black', facecolors = 'none', linewidths=2)
    # edgecolors : 경계선 색, facecolors = 'none' : 속을 비움(색칠 x)



#  등고선을 활용한 결정경계선 출력
#  등고선 출력을 위한 전체 좌표계 생성
X_min = X_petal_len_wd['petal_len'].min() - 0.5
X_max = X_petal_len_wd['petal_len'].max() + 0.5
Y_min = X_petal_len_wd['petal_wd'].min() - 0.5
Y_max = X_petal_len_wd['petal_wd'].max() + 0.5

x1s = np.linspace(X_min, X_max, 100)  # x 좌표 최소~최대 100 균등 분할
y1s = np.linspace(Y_min, Y_max, 100)  # y 좌표 최소~최대 100 균등 분할
print("="*50)
print(x1s)
print(y1s)
# # np.meshgrid(x,y) : x는 아래로 y개수만큼, y는 오른쪽으로 x개수 만큼 펴져서 행렬 데이터 생성
x1, y1 = np.meshgrid(x1s, y1s)
print("="*80)
print(x1.ravel()[:5])
print(len(x1.ravel()))
# x1.ravel() : 모든 요소를 1차원 배열로 펼침(flatten)
# column_stack : 두 x,y 좌표 데이터를 합쳐 전체 좌표계 데이터 생성
xy1 = np.column_stack([x1.ravel(),y1.ravel()])
print("="*80)
print(xy1[:5])
print(len(xy1))
# # 전체 좌표계 데이터로 예측 수행
Z = svm_mm.predict(xy1)
print(Z, Z.shape)
# Z는 1차원 배열로 x1 또는 y1과 동일한 shape로 변형해 주어야 함
Z = Z.reshape(x1.shape)
print(Z.shape)
# # 예측 데이터를 활용한 등고선/색
# # cmap종류 : https://matplotlib.org/stable/tutorials/colors/colormaps.html#
# # Z축에 등고선 높이(Z값)값을 표현하는
# # Z값이 같은 점들끼리 하나의 영역으로 묶이도록 선을 그려줌
# # levels : Z 값 별로  각 리스트의 원소대로 cut
plt.contour(x1, y1, Z, levels=[0.0, 1.0, 2.0], colors = 'red' ) # 등고선 표현
plt.contourf(x1, y1, Z, cmap = plt.cm.RdYlBu, alpha = 0.3 ) # 등고색 표현

plt.xlabel(iris['feature_names'][2])
plt.ylabel(iris['feature_names'][3])
plt.title("IRIS : RBF Kernel( C = {}, gamma = {})".format(c,g))#,np.round(accuracy,3) ) )
plt.legend(loc='best')
plt.savefig('svm_iris_분류.jpeg')