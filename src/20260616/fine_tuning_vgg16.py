from tensorflow.keras.applications import vgg16
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

vgg16model = vgg16.VGG16() # vgg16 모델 네트워크 구조와 가중치까지 모두 로딩2
# vgg16model.summary()

pred_image_path = r'/home/dlgusrb/deeplearning_prg/dataset/testimage_dataset-20260616T081915Z-3-001/testimage_dataset/'

filelist = []
for file in os.listdir(pred_image_path):
    filelist.append(pred_image_path + file)

# print(filelist)

img = load_img(filelist[3], target_size = (224, 224))
image = img_to_array(img) # 이미지 객체를 넘파이 배열로 변환
# print(image.shape) # (224, 224, 3)

image = image.reshape(1, 224, 224, 3) # batch size
# print(image.shape) # (1, 224, 224, 3)

image = vgg16.preprocess_input(image) # 자기 모델에 맞게 정규화, RGB->BGR 변환도 해준다.
# print(image)

pred = vgg16model.predict(image)
# print(len(pred[0])) # batch size때문에 인덱스 들어가야 한다.

# print(np.argmax(pred[0]))

labels = vgg16.decode_predictions(pred)
print(labels)


