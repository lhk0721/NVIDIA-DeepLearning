import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

(train_x, _), (test_x, _) = fashion_mnist.load_data() # 28 * 28 grayscale img
# print(train_x.shape)
# (60000, 28, 28)

train_x = train_x.reshape(-1, 28, 28, 1) / 255.0
# print(train_x.shape)
# (60000, 28, 28, 1)

test_x = test_x.reshape(-1, 28, 28, 1) / 255.0

# print(train_x[0]) # 255.0으로 나누어 정규화

##  모델 준비

model = load_model(r'/home/dlgusrb/deeplearning_prg/models/autoencodermodel.keras')

predict_img = model.predict(test_x)
# print(predict_img.shape)
# print(predict_img[0])
# (10000, 28, 28, 1)
# [[[6.54987480e-06]
#   [8.78220234e-08]


## 결과 확인
import matplotlib.pyplot as plt
num = 5
plt.figure(figsize = (15,7))
for i in range(num):
    ax1 = plt.subplot(2,num, i+1)
    ax1.imshow(test_x[i].reshape(28,28),cmap = 'gray')
    ax1.set_title('original_image %d' %i)
    ax1.axis('off')

    ax1 = plt.subplot(2,num, i+num+1)
    ax1.imshow(predict_img[i].reshape(28,28),cmap = 'gray')
    ax1.set_title('autoenc_imge %d' %i)
    ax1.axis('off')
    
plt.savefig('figures/autoenc.jpeg')