import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Input
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# -------------------------------------------------------------
# [1] 하이퍼파라미터 및 경로 설정
# -------------------------------------------------------------
TRAIN_DIR = r'/home/dlgusrb/deeplearning_prg/dataset/intel image dataset/seg_train'
TEST_DIR  = r'/home/dlgusrb/deeplearning_prg/dataset/intel image dataset/seg_test'

IMG_SIZE = (128, 128)
BATCH_SIZE = 128
EPOCHS = 100

# -------------------------------------------------------------
# [2] 디렉토리로부터 데이터셋 읽어오기 (라벨 없음: 오토인코더)
# -------------------------------------------------------------
raw_train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None,
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
# [3] 전처리: 정규화 + (X, Y) = (images, images)
#     메모리가 넉넉하므로 cache()로 디스크 I/O를 한 번만 발생시켜 학습을 가속
# -------------------------------------------------------------
def preprocess_for_autoencoder(images):
    images = tf.cast(images, tf.float32) / 255.0
    return images, images

train_ds = (raw_train_ds
            .map(preprocess_for_autoencoder, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .shuffle(2000)
            .prefetch(tf.data.AUTOTUNE))
test_ds = (raw_test_ds
           .map(preprocess_for_autoencoder, num_parallel_calls=tf.data.AUTOTUNE)
           .cache()
           .prefetch(tf.data.AUTOTUNE))

print("데이터셋 준비 완료!")

# 데이터 확인(시각화용 — 학습엔 사용 안 함)
for input_images, target_images in train_ds.take(1):
    print(f"입력 배치 크기: {input_images.shape}")
    print(f"최대 픽셀 값: {tf.reduce_max(input_images).numpy()}")
    print(f"입력==타겟: {tf.reduce_all(tf.equal(input_images, target_images)).numpy()}")
    plt.figure(figsize=(3, 3))
    plt.imshow(input_images[0].numpy())
    plt.title("Sample Input Image")
    plt.axis("off")
    plt.savefig(r'figures/intellimg_autoencoder_sample.png')
    plt.close()

# -------------------------------------------------------------
# [4] 강화된 컨볼루션 오토인코더
#     - VGG식 (Conv-Conv-BN) 블록으로 표현력 강화
#     - 디코더는 Conv2DTranspose(학습형 업샘플링)로 선명도 향상
# -------------------------------------------------------------
def conv_block(model, filters):
    model.add(Conv2D(filters, (3, 3), padding='same', activation='leaky_relu'))
    model.add(Conv2D(filters, (3, 3), padding='same', activation='leaky_relu'))
    model.add(BatchNormalization())

def up_block(model, filters):
    # stride 2 transposed conv = 학습 가능한 2배 업샘플링
    model.add(Conv2DTranspose(filters, (3, 3), strides=2, padding='same',
                              activation='leaky_relu'))
    model.add(Conv2D(filters, (3, 3), padding='same', activation='leaky_relu'))
    model.add(BatchNormalization())

model = Sequential()
model.add(Input(shape=(128, 128, 3)))

# --- Encoder ---
conv_block(model, 64);  model.add(MaxPooling2D((2, 2)))   # 64x64
conv_block(model, 128); model.add(MaxPooling2D((2, 2)))   # 32x32
conv_block(model, 256); model.add(MaxPooling2D((2, 2)))   # 16x16

# --- Bottleneck (16x16x256: 공간 정보를 충분히 남겨 선명도 확보) ---
conv_block(model, 256)

# --- Decoder ---
up_block(model, 256)   # 32x32
up_block(model, 128)   # 64x64
up_block(model, 64)    # 128x128

# 출력층: [0,1] 정규화 입력에 맞춰 sigmoid
model.add(Conv2D(3, (3, 3), padding='same', activation='sigmoid'))

model.summary()

# -------------------------------------------------------------
# [5] 컴파일 / 콜백 / 학습
# -------------------------------------------------------------
model.compile(
    loss='mae',                       # mse보다 덜 흐릿한 복원
    optimizer=Adam(learning_rate=1e-3),
    metrics=['mse']
)

checkpoint_cb = ModelCheckpoint(
    filepath='models/autoencoding_intelimg.keras',
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)

earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=12,
    verbose=1,
    restore_best_weights=True
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    verbose=1,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

# -------------------------------------------------------------
# [6] 학습 곡선
# -------------------------------------------------------------
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = np.arange(len(loss))

plt.figure()
plt.plot(epochs, loss, 'b', label='Train_Loss')
plt.plot(epochs, val_loss, 'orange', label='Val_Loss')
plt.legend()
plt.title('Autoencoder Loss (MAE)')
plt.savefig('figures/autoencodermodel_intelimg.jpeg')
plt.close()
