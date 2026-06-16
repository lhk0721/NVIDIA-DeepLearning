import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(train_x, train_y) , (test_x, test_y) = fashion_mnist.load_data()
print(len(train_x), len(test_x)) # 60000, 10000

# 정답의 클래스 분류 와 어떤 클래스가 몇개 있는지?
print( np.unique(train_y , return_counts=True ))
print( np.unique(test_y , return_counts=True ))
# 0 ==> 6000, 1 ==> 6000 ...
# 0 ==> 티셔츠

# train_x ==> (60000, 28, 28)

# import matplotlib.pyplot as plt
# print(train_x[0].shape) # (28, 28)
# # 수치가 0 에 가까운것은  검정색으로
# # 수치가 255에 가까운 것은 흰색으로 표현
# plt.imshow(train_x[0] , cmap='gray')

# plt.savefig('train_0.jpeg')

train_scaled = train_x.reshape(-1,28,28,1) /  255.0  # 이미지 데이터 정규화
print(train_scaled.shape)

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

train_y_encoded = to_categorical(train_y)
print(train_y_encoded)

# train / validata 로 분할
train_x, val_x, train_y, val_y = \
    train_test_split(train_scaled, train_y_encoded , test_size=0.2, random_state=42)

print(len(train_x)) # 48000
print(len(val_x)) # 12000

# 모델 생성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

model = Sequential()
# 첫번째 : 필터의 개수
# kernel_size ==> 3  ==> ( 3 x 3 )
# paddaing = 'same' ==> same padding ==> zero padding 
# 첫번째 층은 항상 입력 데이터를 생각해서 코딩
# input_shape = (28,28,1) ==> 입력 이미지의 Shape
# Conv. layer 층 추가
model.add(  Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                   input_shape = (28,28,1)) )  # Conv lay 층 만들어서 추가 

# 풀링층 추가
model.add( MaxPooling2D(2) ) # 2 ==> 2x2 필터가 2스트라이드 이동하면서 최대값 선택

# Conv. Layer 층 추가
model.add( Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same') )
# 풀링층 추가
model.add( MaxPooling2D(2) ) # 2 ==> 2x2 필터가 2스트라이드 이동하면서 최대값 선택

model.add( Flatten() )
# FC layer 
model.add( Dense(100, activation='relu') )
# 과대 적합 방지
model.add( Dropout(0.4) )
model.add( Dense(40, activation='relu') )
# 출력층 설계 ==> 분류하고자하는 클래스의 수 만큼 뉴런이 필요하다
# fashin_mnist 데이터의 라벨(정답)이 ==> 10개 클래스 분류
# 출력층 ==> 활성화함수 ==> 다중분류 일 경우 ==> softmax
model.add( Dense(10, activation='softmax') )
model.summary()

# 모델 컴파일 ( opimizer, loss , matrics)
model.compile( loss = "categorical_crossentropy", optimizer = 'adam',
              metrics = ['accuracy'] )

# 모델 학습
# 콜백 기능 추가해서 best 모델 저장 , 조기종료
# 조기종료는 val_loss 로 체크해서 조기종료
#model.fit(train_x, train_y, batch_size=64, epochs= 100, verbose = 1)

# callback 기능
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

checkpoint_cb = ModelCheckpoint(filepath='./fashinmnist_model.keras', monitor='val_loss',
                                verbose=1, save_best_only=True)
earlystop_cb = EarlyStopping( monitor='val_loss', patience= 3, restore_best_weights= True)

# 모델 학습
histroy = model.fit(train_x, train_y , validation_data = (val_x, val_y), batch_size = 64, epochs=100, verbose=1,
                callbacks = [checkpoint_cb, earlystop_cb])


