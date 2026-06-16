from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


mnist = datasets.load_digits() # 손글씨 이미지
features = mnist['data'] # 입력데이터
print(len(features[0]))
print(len(features)) # 1797

labels = mnist['target'] # 라벨데이터
print(len(labels)) # 1797
print(np.unique(labels, return_counts=True))

# print(features[3].reshape(8, 8))
# plt.imshow(features[3].reshape(8, 8), cmap='gray')
# plt.savefig('mnist_0.jpeg')

print(features.shape)
# ==> (1797, 8, 8, 1) ==> (batch_size, 이미지 가로, 이미지 세로, 채널(깊이))
features = features.reshape(-1, 8, 8, 1) / 255.0 # / 255.0 ==> 사이즈 변경 + 스케일 정규화


# features와 label을 train_x, val_x로 분할 (분할비율 0.2)
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size=0.2, random_state=42)
print(len(train_x))
print(len(val_x))
# ==> 데이터 전처리 및 데이터 준비 완료


# 모델 준비 ==> 이미지 분류 모델 설계 (10개 클래스 분류: 다중분류)
# 이미지 분류 특화 모델 ==> cnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Flatten

mnist_model = Sequential()

# 2개의 Conv, 2개의 Pooling
mnist_model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(8, 8, 1), activation='leaky_relu'))
mnist_model.add(MaxPooling2D(2, 2))
# mnist_model.summary()

mnist_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='leaky_relu'))
mnist_model.add(MaxPooling2D(2, 2))
# mnist_model.summary()

# flatten, drop-out, FC layer 층 추가
mnist_model.add(Flatten())
# mnist_model.summary()
mnist_model.add(Dense(units=100, activation='leaky_relu'))
mnist_model.add(Dropout(0.3))
mnist_model.add(Dense(units=60, activation='leaky_relu'))
# mnist_model.summary()

# # 마지막 출력층 10개 뉴런으로 설정
mnist_model.add(Dense(units=10, activation='softmax'))
mnist_model.summary()

# 손실함수 ==> categorical_cossentropy
# target(정답) 정수형태 그대로 사용 ==> sparse_categorical_crossentropy
# categorical_crossentropy ==> 원핫인코딩 상태로 변경해서 전달  
mnist_model.compile(loss='sparse_categorical_crossentropy', 
                    optimizer='adam', metrics = ['accuracy'])

# 모델 설계 후 모델 학습
# val_x, val_y 전달해서 val_loss 모니터링으로 조기종료 콜백
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
modelpath='./20260615/mnist_bestmodel.keras'
modelcheck_cb = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0,save_best_only=True)
earlystopping_cb = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = mnist_model.fit(train_x, train_y, validation_data=[val_x, val_y], batch_size=4, epochs=100, verbose=1,
                callbacks=[modelcheck_cb, earlystopping_cb])

# mnist_model.save('./20260615/mnist_model.keras')