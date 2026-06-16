import numpy as np
import pandas as pd

# e 지수 표현하는 과학적표기 대신  소수점 이하 8자리까지 표현
np.set_printoptions(precision=8, suppress=True)
np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력

fishdf = pd.read_csv('/home/sckit/deeplearning_prj/20260611/fish_data.csv')
print(fishdf)
print(fishdf['Species'].unique()) # 물고기의 종이 몇종? ( 총 7종의 물고기 )
# 7개 물고기 중 어떤 물고기야 ?  ( 다중분류 )
fishdf.info()

# Series 를 넘파이 배열로 변환해주는 메서드 ==> to_numpy()
fish_target = fishdf['Species'].to_numpy()
print(fish_target)
print(fishdf.columns)
fish_train = fishdf[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_train)

# 라벨(문자열)을 수치형태로 변환 ==> Labelencoder()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(fish_target)
print(y_encoded) # 타깃이 수치형태로 변환됨 ==> 1, 2, 3, 4

print(le.classes_)

# categorical_crossentropy() ==> 정답이 원-핫인코딩 상태이어햐함
from tensorflow.keras.utils import to_categorical
y_onehot = to_categorical(y_encoded)
print(y_onehot)

# train /test 데이터 분리
from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = \
    train_test_split(fish_train, y_onehot, random_state=42)

print(train_x.shape)
print(val_x.shape)

# 특성 데이터한 스케일 조정 ( 표준점수 정규화 )
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_x)
test_scaled = scaler.transform(val_x)

import joblib
joblib.dump(scaler, 'fish_scaler.pkl')
#  반대로 읽어들이떄는  joblib.load('fish_scaler.pkl')

# 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

multi_model = Sequential()
multi_model.add( Dense(units=10, input_dim=5, activation='leaky_relu') )
multi_model.add( Dense(units=7, activation='softmax')  )

#multi_model.summary()

multi_model.compile(loss='categorical_crossentropy', optimizer='adam',
                    metrics=['accuracy'])
# val 데이터를 가지고 loss 개선됬을때 모델을 저장하고 개선되지 않을 경우 그냥 넘어가라
# callback 기능
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

checkpoint_cb = ModelCheckpoint(filepath='./fish_bestmodel.keras', monitor='val_loss',
                                verbose=1, save_best_only=True)
earlystop_cb = EarlyStopping( monitor='val_loss', patience= 3, restore_best_weights= True)

# 모델 학습
multi_model.fit(train_scaled, train_y , validation_data = (test_scaled, val_y), batch_size = 1, epochs=500, verbose=1,
                callbacks = [checkpoint_cb, earlystop_cb])

# # 모델 성능평가
# print('Test acc : ', multi_model.evaluate(test_scaled, test_y)[1])

# multi_model.save('fish_multi_clf.keras')