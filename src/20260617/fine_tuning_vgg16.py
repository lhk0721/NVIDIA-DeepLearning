from tensorflow.keras.applications import vgg16
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


##

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = r'/home/dlgusrb/deeplearning_prg/dataset/Covid19-dataset/train'
test_dir = r'/home/dlgusrb/deeplearning_prg/dataset/Covid19-dataset/test'
batch_size = 4
image_size = 224


## train 이미지 증강 생성
train_image_generator = ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    vertical_flip = False,
)

## test
test_image_generator = ImageDataGenerator()

## train image를 읽어들이면서 이미지를 증강시켜주는 generator 생성
train_data_gen = train_image_generator.flow_from_directory(
    train_dir, 
    batch_size = batch_size,
    shuffle = True,
    class_mode = 'categorical',
    target_size = (image_size,image_size)
)

## test image를 읽어들이면서 이미지를 증강시켜주는 generator 생성
test_data_gen = test_image_generator.flow_from_directory(
    test_dir, 
    batch_size = batch_size,
    shuffle = True,
    class_mode = 'categorical', 
    target_size = (image_size,image_size)
)

class_num = len(train_data_gen.class_indices)
# print(class_num,train_data_gen.class_indices)
# 2 {'Covid': 0, 'Normal': 1}

class_labels = list(test_data_gen.class_indices.keys())
# print(class_labels[0],class_labels[1])
# Covid Normal

## vgg16 모델의 가중치를 가져와서 전이학습하는 모델 설계

vgg16model = vgg16.VGG16(
    weights='imagenet',
    include_top = False, # fclayer는 재설계할 것임.
    input_shape = (image_size,image_size,3)
) 

vgg16model.summary()

for layer in vgg16model.layers:
    layer.trainable = False # vgg16의 특징추출 layer는 학습되지 마세요

newmodel = Sequential()
newmodel.add(vgg16model)

newmodel.add(Flatten())
newmodel.add(Dense(1024, activation='relu'))
newmodel.add(Dropout(0.3))
newmodel.add(Dense(class_num, activation='softmax'))

newmodel.summary()

newmodel.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

import numpy as np
# int(np.ceil( 181 // 4 )) ## step size 45면 하나 남으니까 ceil해서 더해준다.
# print(train_data_gen.samples)
# print(test_data_gen.batch_size)

# print(int(np.ceil( train_data_gen.samples // test_data_gen.batch_size )))

## 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

checkpoint_cb = ModelCheckpoint(
    filepath='models/covid19.keras',
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)

earlystop_cv = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

newmodel.fit(
    train_data_gen,
    steps_per_epoch = int(np.ceil( train_data_gen.samples / train_data_gen.batch_size )),
    epochs = 25,
    validation_data = test_data_gen,
    validation_steps = int(np.ceil( test_data_gen.samples / test_data_gen.batch_size )),
    verbose = 1,
    callbacks = [checkpoint_cb, earlystop_cv]
)

## 
