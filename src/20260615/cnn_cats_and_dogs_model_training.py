train_dir = r'dataset/cats_and_dogs/training_set'
test_dir = r'dataset/cats_and_dogs/test_set'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

## train 이미지 증강 생성
train_image_generator = ImageDataGenerator(
    rescale = 1.0/255.,
    rotation_range = 20,
    height_shift_range = 2.0
)

## test는 scale 변환만 해준다.
test_image_generator = ImageDataGenerator(
    rescale = 1.0/255.,
)

## train image를 읽어들이면서 이미지를 증강시켜주는 generator 생성
train_data_gen = train_image_generator.flow_from_directory(
    train_dir, # 불러올 이미지 경로
    batch_size = 20,
    shuffle = True,
    # 디렉토리 내부 이미지를 읽어올 때 어떤 형식으로 라벨링해서 불러올 것이냐? -> classmode
    class_mode = 'binary', # 이진분류, sigmoid. 다중분류 때는 'categorical', 활성화함수를 softmax로 줘야 함
    target_size = (150, 150) # cnn모델 입력 사이즈로 리사이즈해라
)

## test image를 읽어들이면서 이미지를 증강시켜주는 generator 생성
test_data_gen = test_image_generator.flow_from_directory(
    test_dir, # 불러올 이미지 경로
    batch_size = 20,
    shuffle = True,
    # 디렉토리 내부 이미지를 읽어올 때 어떤 형식으로 라벨링해서 불러올 것이냐? -> classmode
    class_mode = 'binary', # 이진분류, sigmoid. 다중분류 때는 'categorical', 활성화함수를 softmax로 줘야 함
    target_size = (150, 150) # cnn모델 입력 사이즈로 리사이즈해라
)

## 모델 준비
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(
    Conv2D(
        16,
        kernel_size=(3, 3), 
        input_shape=(150, 150, 3), 
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=2
    )
)

model.add(
    Conv2D(
        32, 
        (3, 3), 
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=2
    )
)

model.add(
    Conv2D(
        64, 
        (3, 3), 
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=2
    )
)

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

## 조기종료

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint_cb = ModelCheckpoint(
    'models/catdof_bestmodel.keras',
    save_best_only=True
)

earlystopping_cb = EarlyStopping(
    patience=3,
    restore_best_weights=True
)

##모델 학습(generator를 활용)

history = model.fit(
    train_data_gen,
    validation_data = test_data_gen,
    steps_per_epoch = 400, # 총 traindata / batchsize 20
    validation_steps = 100,
    verbose = 1,
    epochs = 50,
    callbacks = [checkpoint_cb, earlystopping_cb]
)

## 성능 시각화

import matplotlib.pyplot as plt

acc = history.history['accuracy'] # train data의 정확도
val_acc = history.history['val_accuracy'] # test 데이터의 정확도
loss = history.history['loss'] # test 데이터의 정확도
val_loss = history.history['val_loss'] # test 데이터의 정확도

epochs = range(len(acc))
plt.figure()

plt.plot(
    epochs,
    loss,
    'go',
    label = 'Train loss'
)

plt.plot(
    epochs,
    val_loss,
    'g',
    label = 'Val loss'
)

plt.legend()
plt.savefig('figures/cats_and_dogs_model.jpeg')

## 


# from tensorflow.keras.models import load_model
# best_cnn_model = load_model(r'src/20260615/')

