from sklearn import datasets
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataSet = datasets.load_digits() # 0~9로 이루어진 손글씨 데이터셋

# print(dataSet.data)
# [[ 0.  0.  5. ...  0.  0.  0.]
#  [ 0.  0.  0. ... 10.  0.  0.]
#  [ 0.  0.  0. ... 16.  9.  0.]
#  ...
#  [ 0.  0.  1. ...  6.  0.  0.]
#  [ 0.  0.  2. ... 12.  0.  0.]
#  [ 0.  0. 10. ... 12.  1.  0.]]

# print(dataSet.target)
# [0 1 2 ... 8 9 8]

features = dataSet['data']
# print(len(features))
# 1797

labels = dataSet['target']
# print(len(labels))
# 1797

train_x, test_x, train_y, test_y =\
    train_test_split(
        features, labels, test_size=0.2
)

##

# tree
model_tree = tree.DecisionTreeClassifier(
    criterion='gini',
    max_depth=8,
    random_state=46,
    max_features=32
)

model_tree.fit(train_x,train_y)

# knn
model_knn = KNeighborsClassifier(n_neighbors=299)

model_knn.fit(
    X=train_x,
    y=train_y
)

# svc
model_svc = SVC(
    C=0.1,
    gamma=0.003, 
    probability=True, 
    random_state=46
)

model_svc.fit(train_x,train_y)

model_hardVoting = VotingClassifier(
    estimators=[
        ('decision_tree', model_tree),
        ('knn',model_knn),
        ('svm',model_svc)
    ], 
    weights=[1,1,1],
    voting='hard'
)

model_hardVoting.fit(train_x,train_y)


model_softVoting = VotingClassifier(
    estimators=[
        ('decision_tree', model_tree),
        ('knn',model_knn),
        ('svm',model_svc)
    ], 
    weights=[1,1,1],
    voting='soft'
)

model_softVoting.fit(train_x,train_y)

# print('='*80)
# print('tree acc: ', model_tree.score(test_x,test_y))
# print('='*80)
# print('knn acc: ',model_knn.score(test_x,test_y))
# print('='*80)
# print('svc acc: ',model_svc.score(test_x,test_y))
# print('='*80)
# print('hard voting acc: ',model_hardVoting.score(test_x,test_y))
# print('='*80)
# print('soft voting acc: ',model_softVoting.score(test_x,test_y))
# print('='*80)

# ================================================================================
# tree acc:  0.8583333333333333
# ================================================================================
# knn acc:  0.8472222222222222
# ================================================================================
# svc acc:  0.9416666666666667
# ================================================================================
# hard voting acc:  0.9305555555555556
# ================================================================================
# soft voting acc:  0.9083333333333333
# ================================================================================



