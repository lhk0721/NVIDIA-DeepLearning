import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# [1] 하이퍼파라미터 및 경로 설정
# -------------------------------------------------------------
# TODO: 본인의 Intel Image Classification 데이터셋 경로에 맞게 수정하세요.
TRAIN_DIR = r'home/dlgusrb/deeplearning_prg/dataset/intel image dataset/seg_train'
TEST_DIR = r'/home/dlgusrb/deeplearning_prg/dataset/intel image dataset/seg_test'

IMG_SIZE = (64, 64)
BATCH_SIZE = 64

# -------------------------------------------------------------
# [2] 디렉토리로부터 데이터셋 읽어오기 (기본은 라벨 포함)
# -------------------------------------------------------------
raw_train_ds = image_dataset_from_directory(
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# [1] 하이퍼파라미터 및 경로 설정
# -------------------------------------------------------------
# TODO: 본인의 Intel Image Classification 데이터셋 경로에 맞게 수정하세요.
TRAIN_DIR = r'/home/oys0117/deeplearning_prj/20260615/dataset2/seg_train'
TEST_DIR = r'/home/oys0117/deeplearning_prj/20260615/dataset2/seg_test'

IMG_SIZE = (64, 64)
BATCH_SIZE = 64

# -------------------------------------------------------------
# [2] 디렉토리로부터 데이터셋 읽어오기 (기본은 라벨 포함)
# -------------------------------------------------------------
raw_train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None, # Autoencoder이므로 분류용 라벨은 로드하지 않음 (오직 이미지 데이터만)
    shuffle=True
)

raw_test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None,
    shuffle=False
)

# -------------------------------------------------------------
# [3] Autoencoder를 위한 전처리 파이프라인 (정규화 및 X=Y 매핑)
# -------------------------------------------------------------
def preprocess_for_autoencoder(images):
    # 1. 픽셀 값을 [0, 255]에서 [0.0, 1.0] 범위로 정규화
    images = tf.cast(images, tf.float32) / 255.0
    # 2. ★ 핵심: 입력을 그대로 정답(Target)으로 반환 (X, Y) -> (images, images)
    return images, images

# .map()을 이용해 전체 데이터셋에 전처리 적용
# tf.data.AUTOTUNE을 주어 성능을 최적화합니다.
train_ds = raw_train_ds.map(preprocess_for_autoencoder, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = raw_test_ds.map(preprocess_for_autoencoder, num_parallel_calls=tf.data.AUTOTUNE)

# 메모리 버퍼링 및 프리페치 설정 (학습 속도 향상)
train_x = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_x = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("데이터셋 준비 완료!")

# 2. 데이터 로딩 확인 및 시각화

# 하나의 배치를 가져와서 형태 확인
for input_images, target_images in train_ds.take(1):
    print(f"입력 배치 크기: {input_images.shape}")   # (64, 64, 64, 3)
    print(f"타겟 배치 크기: {target_images.shape}")   # (64, 64, 64, 3)
    print(f"최대 픽셀 값: {tf.reduce_max(input_images).numpy()}") # 1.0 내외인지 확인

    # 입력과 타겟이 완벽히 똑같은지 한 번 더 확인
    is_identical = tf.reduce_all(tf.equal(input_images, target_images)).numpy()
    print(f"입력과 타겟이 완벽히 일치합니까?: {is_identical}")

    # 샘플 이미지 한 장 띄워보기
    plt.figure(figsize=(3, 3))
    plt.imshow(input_images[0].numpy())
    plt.title("Sample Input Image")
    plt.axis("off")
    plt.show()



#------모델 설계-------#

autoencoder_model = Sequential()
#ecoder 부분
autoencoder_model.add(Input(shape=(64, 64, 3))) #배치 사이즈를 제외한 원 입력이미지 shape을 전달
autoencoder_model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same',
                             activation='leaky_relu'))
autoencoder_model.add(MaxPooling2D(pool_size=(2,2)))
autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='leaky_relu'))
autoencoder_model.add(MaxPooling2D(pool_size=(2,2)))
autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3), strides=2, padding='same', activation='leaky_relu'))

#decoder 부분
autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())

autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())

autoencoder_model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())

autoencoder_model.add(Conv2D(filters=3, kernel_size=(3,3), padding='same', activation='linear'))

autoencoder_model.summary()



#======모델 훈련 및 loss 출력======#

import tensorflow as tf

#adam의 default lr은 0.001

# 조기종료 코드
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
ckpt = ModelCheckpoint('./intel_autoencoder.keras', save_best_only=True)
early = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
autoencoder_model.compile(optimizer=optimizer, loss='mse')
history = autoencoder_model.fit(train_x, epochs=50, batch_size=64,
                                validation_data=test_x, callbacks=[ckpt, early])

# # trainloss = history.history['loss']
# # valloss = history.history['val_loss']
# # import matplotlib.pyplot as plt

# # plt.plot(trainloss)
# # plt.plot(valloss)
# # plt.savefig('autoencoder_mse.jpeg')TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None, # Autoencoder이므로 분류용 라벨은 로드하지 않음 (오직 이미지 데이터만)
    shuffle=True
)

raw_test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None,
    shuffle=False
)

# -------------------------------------------------------------
# [3] Autoencoder를 위한 전처리 파이프라인 (정규화 및 X=Y 매핑)
# -------------------------------------------------------------
def preprocess_for_autoencoder(images):
    # 1. 픽셀 값을 [0, 255]에서 [0.0, 1.0] 범위로 정규화
    images = tf.cast(images, tf.float32) / 255.0
    # 2. ★ 핵심: 입력을 그대로 정답(Target)으로 반환 (X, Y) -> (images, images)
    return images, images

# .map()을 이용해 전체 데이터셋에 전처리 적용
# tf.data.AUTOTUNE을 주어 성능을 최적화합니다.
train_ds = raw_train_ds.map(preprocess_for_autoencoder, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = raw_test_ds.map(preprocess_for_autoencoder, num_parallel_calls=tf.data.AUTOTUNE)

# 메모리 버퍼링 및 프리페치 설정 (학습 속도 향상)
train_x = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_x = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("데이터셋 준비 완료!")

# 2. 데이터 로딩 확인 및 시각화

# 하나의 배치를 가져와서 형태 확인
for input_images, target_images in train_ds.take(1):
    print(f"입력 배치 크기: {input_images.shape}")   # (64, 64, 64, 3)
    print(f"타겟 배치 크기: {target_images.shape}")   # (64, 64, 64, 3)
    print(f"최대 픽셀 값: {tf.reduce_max(input_images).numpy()}") # 1.0 내외인지 확인

    # 입력과 타겟이 완벽히 똑같은지 한 번 더 확인
    is_identical = tf.reduce_all(tf.equal(input_images, target_images)).numpy()
    print(f"입력과 타겟이 완벽히 일치합니까?: {is_identical}")

    # 샘플 이미지 한 장 띄워보기
    plt.figure(figsize=(3, 3))
    plt.imshow(input_images[0].numpy())
    plt.title("Sample Input Image")
    plt.axis("off")
    plt.show()



#------모델 설계-------#

autoencoder_model = Sequential()
#ecoder 부분
autoencoder_model.add(Input(shape=(64, 64, 3))) #배치 사이즈를 제외한 원 입력이미지 shape을 전달
autoencoder_model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same',
                             activation='leaky_relu'))
autoencoder_model.add(MaxPooling2D(pool_size=(2,2)))
autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='leaky_relu'))
autoencoder_model.add(MaxPooling2D(pool_size=(2,2)))
autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3), strides=2, padding='same', activation='leaky_relu'))

#decoder 부분
autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())

autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())

autoencoder_model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())

autoencoder_model.add(Conv2D(filters=3, kernel_size=(3,3), padding='same', activation='linear'))

autoencoder_model.summary()



#======모델 훈련 및 loss 출력======#

import tensorflow as tf

#adam의 default lr은 0.001

# 조기종료 코드
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
ckpt = ModelCheckpoint('./intel_autoencoder.keras', save_best_only=True)
early = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
autoencoder_model.compile(optimizer=optimizer, loss='mse')
history = autoencoder_model.fit(train_x, epochs=50, batch_size=64,
                                validation_data=test_x, callbacks=[ckpt, early])

# # trainloss = history.history['loss']
# # valloss = history.history['val_loss']
# # import matplotlib.pyplot as plt

# # plt.plot(trainloss)
# # plt.plot(valloss)
# # plt.savefig('autoencoder_mse.jpeg')