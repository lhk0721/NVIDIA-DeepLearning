
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0,10, 10)
print(X)
print(X.shape, type(X.shape)) # (10,)

Y = X + np.random.randn(*X.shape)
print(Y)

# 딥러닝 선형회귀 모델 설계
# 입력데이터(x) 는 1개
# 입력층 의 뉴런은 1개
# 출력층 의 활성화는 lineer
# 손실함수 : MSE
linear_model = Sequential()  # 모델 설계 틀 준비

#linear_model.add(Input(shape=(1,)))
linear_model.add( Dense(units=1, input_dim = 1, activation='linear', use_bias=False)  )
linear_model.summary()

# # 모델 사용할 준비과정 ==> 환경설정과정
# linear_model.compile(loss="mse", optimizer="adam", metrics=["mae"])

# # 학습전의 가중치(w) 체크 
# wights = linear_model.layers[0].get_weights()
# w = wights[0][0][0]
# print('fit 전 가중치 체크 : ', w)
# # 학습(fit)

# linear_model.fit(X, Y, batch_size = 1, epochs = 1000, verbose = 1)

# # 학습완료 후의 가중치(w) 체크 
# wights = linear_model.layers[0].get_weights()
# w = wights[0][0][0]
# print('fit 완료 후 가중치 체크 : ', w)

# plt.plot(X, Y, label='data')
# # 모델이 찾은 선형회귀 선을 표시
# plt.plot(X, w*X , label = 'pred')
# plt.legend() # label을 차트에 뿌려라
# plt.savefig('figures/linear_model.jpeg')





