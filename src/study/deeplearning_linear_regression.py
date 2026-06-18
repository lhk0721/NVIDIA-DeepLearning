import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#농어길이데이터 ( 캐글Fish Market 데이터참조)
perch_length=np.array([8.4,13.7,15.0,16.2,17.4,18.0,18.7,19.0,19.6,20.0,21.0,
21.0,21.0,21.3,22.0,22.0,22.0,22.0,22.0,22.5,22.5,22.7,
23.0,23.5,24.0,24.0,24.6,25.0,25.6,26.5,27.3,27.5,27.5,
27.5,28.0,28.7,30.0,32.8,34.5,35.0,36.5,36.0,37.0,37.0,
39.0,39.0,39.0,40.0,40.0,40.0,40.0,42.0,43.0,43.0,43.5,
44.0])
# 농어무게데이터 (캐글FishMarket 데이터참조)
perch_weight=np.array([5.9,32.0,40.0,51.5,70.0,100.0,78.0,80.0,85.0,85.0,110.0,
115.0,125.0,130.0,120.0,120.0,130.0,135.0,110.0,130.0,
150.0,145.0,150.0,170.0,225.0,145.0,188.0,180.0,197.0,
218.0,300.0,260.0,265.0,250.0,250.0,300.0,320.0,514.0,
556.0,840.0,685.0,700.0,700.0,690.0,900.0,650.0,820.0,
850.0,900.0,1015.0,820.0,1100.0,1000.0,1100.0,1000.0,
1000.0])

## 전처리
(
    train_x,
    test_x,
    train_y,
    test_y
) = train_test_split(
    perch_length,
    perch_weight,
    random_state=42
)

# print(train_x.shape, test_x.shape)
# (42,) (14,)

train_x = train_x.reshape(-1,1)
test_x = test_x.reshape(-1,1)

# print(train_x.shape, test_x.shape)
# (42, 1) (14, 1)

## 특성 추가
train_poly = np.column_stack((train_x**2, train_x))
test_poly = np.column_stack((test_x**2, test_x))

# print(train_poly.shape, test_poly.shape)
# (42, 2) (14, 2)

## 모델 설계 딥러닝 선형회귀
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

multi_model = Sequential()

# 4개 뉴런 입력층
multi_model.add(
    Dense(
        units=4,
        input_dim = 2,
        activation = 'leaky_relu'
    )
)

# 8개 뉴런 은닉층
multi_model.add(
    Dense(
        units=8,
        activation='leaky_relu'
    )
)

# 출력 뉴런 1
multi_model.add(
    Dense(
        units=1,
        activation='linear'
    )
)

# 모델 summary
multi_model.summary()

# model compile
multi_model.compile(
    loss = 'mse',
    optimizer = 'adam',
    metrics = ['mae']
)

## 모델 학습

multi_model.fit(
    train_poly,
    train_y,
    batch_size = 1,
    epochs = 500,
    verbose = 1
)

## 결과
print(multi_model.evaluate(test_poly, test_y))

pred = multi_model.predict(train_poly[:5])
print(pred)
print('='*80)
print(train_y[:5])

## 시각화
# plt.scatter(pred,train_y[:5])
# plt.savefig('figures/deeplearning_multi_fishData.jpeg')