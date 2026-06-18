from tensorflow.keras.models import load_model
from tensorflow.keras.applications import vgg16

newmodel = load_model(r'models/covid19.keras')
newmodel.summary()

##
from tensorflow.keras.preprocessing import image
import numpy as np
class_list = ['covid19', 'normal']

def pred_vgg16_newmodel(newmodel, filename):
    img = image.load_img(filename, target_size=(224,224))
    img_arr = image.img_to_array(img)
    image_reshape = img_arr.reshape((1,224,224,3))
    image_input = vgg16.preprocess_input(image_reshape)

    # Covid: 0, normal: 1

    pred = newmodel.predict(image_input,batch_size = 1)
    print(pred)
    
    print('pred result: ', class_list[np.argmax(pred)])
    # pred_list.append( class_list[np.argmax(pred)])
    return class_list[np.argmax(pred)]

##
import os

test_dir = r'/home/dlgusrb/deeplearning_prg/dataset/Covid19-dataset/test/Covid/'
fileNameList = os.listdir(test_dir)
# print(fileNameList)

file_totalinfo = []
file_list = []

for file in fileNameList:
    file_totalinfo.append(test_dir+file)
    file_list.append(file)
    
print(file_totalinfo)

pred_resultList = []

for imagefile in file_totalinfo:
    pred_resultList.append(pred_vgg16_newmodel(newmodel,imagefile))


##

class_name = 'covid19'

import pandas as pd

from sklearn.metrics import accuracy_score

df = pd.DataFrame({
        'True_Data': [class_name]*len(file_totalinfo),
        'Pred_Data': pred_resultList,
        'FileName': file_list
    })

# print(df)

df.insert(2, 'Result',df['True_Data'] == df['Pred_Data'])

df['Result'] = df['Result'].map(lambda x : 'Positive' if x == True else False)
print(df)

print('accuracy : %.3f' %accuracy_score(df['True_Data'], df['Pred_Data']))