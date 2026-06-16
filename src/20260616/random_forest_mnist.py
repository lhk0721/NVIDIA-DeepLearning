from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

mnist = load_digits()

features = mnist['data']
labels = mnist['target']

train_x, test_x, train_y, test_y = \
    train_test_split(
        features, labels, test_size=0.2
    )

RFmodel = RandomForestClassifier()
RFmodel.fit(train_x,train_y)
# print('score: ',RFmodel.score(test_x,test_y))
# score:  0.9888888888888889



tempdata = [ 0. , 0. ,10. ,14. , 8. , 1. , 0. ,0. ,0. ,2. ,16. ,14. ,6. ,15. , 6. ,0. , 0. , 0., 12. ,15. , 8. ,15. , 0. , 0. , 0. , 15. ,5. ,16. ,16. ,10. ,0. ,0. ,0. ,0. ,12. ,15., 13. ,12. , 0. , 0. , 0. , 4. ,16. , 5. ,4. ,16. ,6. ,0. ,0. ,8. ,16. ,10. ,8. ,16., 8. , 1. , 0. , 1. , 7. ,12. ,14. ,12. ,1. ,0.]

temparr = np.array(tempdata)

temp_pred = RFmodel.predict([temparr])
print('temp_pred: ', temp_pred)
plt.imshow(temparr.reshape(8,8), cmap='gray')

plt.savefig('random_forest_predict.jpeg')