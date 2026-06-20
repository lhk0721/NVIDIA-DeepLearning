import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

(train_x, _), (test_x, _) = fashion_mnist.load_data() # 28 * 28 grayscale img
# print(train_x.shape)
# (60000, 28, 28)

train_x = train_x.reshape(-1, 28, 28, 1) / 255.0
# print(train_x.shape)
# (60000, 28, 28, 1)

test_x = test_x.reshape(-1, 28, 28, 1) / 255.0

print(train_x[0]) # 255.0으로 나누어 정규화

##  모델 준비

## encoder

model = Sequential()

model.add(Input(shape = (28,28,1)))

model.add(
    Conv2D(
        filters=16,
        kernel_size=(3,3),
        padding='same',
        activation = 'leaky_relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size = (2,2)
    )
)

model.add(
    Conv2D(
        filters=8,
        kernel_size=(3,3),
        padding='same',
        activation = 'leaky_relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size = (2,2)
    )
)

model.add(
    Conv2D( # 잠재 벡터
        filters=8,
        kernel_size=(3,3),
        strides=2,
        padding='same',
        activation = 'leaky_relu'
    )
)

## decoder

model.add(
    Conv2D(
        filters=8,
        kernel_size=(3,3),
        padding='same',
        activation = 'leaky_relu'
    )
)

model.add(
    UpSampling2D()
)

model.add(
    Conv2D(
        filters=8,
        kernel_size=(3,3),
        padding='same',
        activation = 'leaky_relu'
    )
)

model.add(
    UpSampling2D()
)

model.add(
    Conv2D(
        filters=16,
        kernel_size=(3,3),
        padding='valid',
        activation = 'leaky_relu' 
    )
)

model.add(
    UpSampling2D()
)

model.add(
    Conv2D(
        filters=1,
        kernel_size=(3,3),
        padding='same',
        activation = 'sigmoid'
    )
)

model.summary()

##

# adam -> 디폴트 학습률: 0.001 (1e-3)

model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

checkpoint_cb = ModelCheckpoint(
    filepath='models/simplernn_imdb.keras',
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)

earlystop_cv = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

history = model.fit(
    train_x, train_x,
    epochs = 50,
    validation_data = (test_x, test_x),
    verbose = 1,
    batch_size = 256,
    callbacks = [checkpoint_cb, earlystop_cv]
)



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt
epochs = np.arange(len(acc))

plt.figure()

plt.plot(
    epochs,
    loss,
    'b',
    label = 'Train_Loss'
)

plt.plot(
    epochs,
    val_loss,
    'orange',
    label = 'Val_Loss'
)

plt.legend()
plt.savefig('figures/autoencodermodel.jpeg')



model.save('models/autoencodermodel.keras') # epoch 모두 끝난 후 저장