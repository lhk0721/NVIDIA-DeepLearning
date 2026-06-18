from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


# 과학적 표기법 대신 소수점 3자리까지 나타냄
np.set_printoptions(precision=3, suppress=True)


mnist = datasets.load_digits() # 손글씨 이미지
features = mnist['data'] # 입력데이터
print(len(features[0]))
print(len(features)) # 1797

labels = mnist['target'] # 라벨데이터
print(len(labels)) # 1797
print(np.unique(labels, return_counts=True))

print(features[0].reshape(8, 8))

print(features.shape)
# ==> (1797, 8, 8, 1) ==> (batch_size, 이미지 가로, 이미지 세로, 채널(깊이))
features = features.reshape(-1, 8, 8, 1) / 255.0 # / 255.0 ==> 사이즈 변경 + 스케일 정규화

# features와 label을 train_x, val_x로 분할 (분할비율 0.2)
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size=0.2, random_state=42)
print(len(train_x))
print(len(val_x))

print(val_x[0]) # predict
print(val_y[0]) # 6

from tensorflow.keras.models import load_model
mnistmodel = load_model('models/mnist_bestmodel.keras')
mnistmodel.summary()

# shape 달라서 오류 ==> 슬라이싱 문법으로 변경
pred = mnistmodel.predict((val_x[0:1]))
print(val_x[0])
print(np.argmax(pred, axis=1))

