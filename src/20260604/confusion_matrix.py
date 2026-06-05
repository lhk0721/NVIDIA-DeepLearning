from sklearn.metrics import classification_report # 정확도, 정밀도 출력
from sklearn.metrics import confusion_matrix # 혼돈행렬 생성


## 이진 분류의 경우
# y_true = [0, 1, 1, 1, 0, 1, 0]
# y_pred = [0, 1, 1, 0, 1, 1, 1]

# print(confusion_matrix(y_true, y_pred)) # 혼돈행렬 (배열) 출력
# 0부터 시작.(0,1) / (0,1)
# [[1 2]
#  [1 3]]

# print(classification_report(y_true, y_pred)) # 정확도, 정밀도 출력
#               precision    recall  f1-score   support

#            0       0.50      0.33      0.40         3
#            1       0.60      0.75      0.67         4

#     accuracy                           0.57         7
#    macro avg       0.55      0.54      0.53         7
# weighted avg       0.56      0.57      0.55         7

## 다중 분류의 경우
y_true = [0, 1, 1, 1, 0, 1, 0,2,2,3,3,3]
y_pred = [0, 1, 1, 0, 1, 1, 1,2,1,3,3,2]

# print(confusion_matrix(y_true, y_pred)) 
# [[1 2 0 0]
#  [1 3 0 0]
#  [0 1 1 0]
#  [0 0 1 2]]

# print(classification_report(y_true, y_pred))
#               precision    recall  f1-score   support

#            0       0.50      0.33      0.40         3
#            1       0.50      0.75      0.60         4
#            2       0.50      0.50      0.50         2
#            3       1.00      0.67      0.80         3

#     accuracy                           0.58        12
#    macro avg       0.62      0.56      0.57        12
# weighted avg       0.62      0.58      0.58        12