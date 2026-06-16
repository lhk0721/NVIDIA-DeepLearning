
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# 소숫점 이하 3자리까지 출력
np.set_printoptions(precision=8, suppress=True)

fish_bestmodel = load_model('fish_multi_clf.keras')
fish_bestmodel.summary()

newfish_scaler = joblib.load('fish_scaler.pkl')


newfishdata = [ [5, 6, 7, 8, 9],
                [20, 30, 20, 30, 20], [55, 35, 25, 45, 45] ]

newfishdata_scaled = newfish_scaler.transform(newfishdata)
print(newfishdata_scaled)
pred = fish_bestmodel.predict(newfishdata_scaled)
print(pred)
print ( np.argmax(pred, axis=1) )  # 최대값의 인덱스로 이루어진 넘파이 배열

fishclass = np.array(['Bream','Parkki','Perch','Pike','Roach','Smelt','Whitefish'])

print( fishclass[np.argmax(pred, axis=1)] )