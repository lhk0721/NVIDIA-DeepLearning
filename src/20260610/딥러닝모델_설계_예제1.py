
import tensorflow as tf
from tensorflow.keras.layers import Dense  # 새로운 층 만들때 사용
from tensorflow.keras.models import Sequential


model = Sequential()  # 딥러닝 층을 추가할수 있는 전체 틀 생성
# Dense()의 첫파라미터는 해당 층의 뉴런의 개수를 지정
model.add(Dense(1,input_dim = 3, activation='sigmoid'))

# 모델이 잘 설계 됬는지 체크
model.summary()