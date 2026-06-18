import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(_, _) , (test_x, _) = fashion_mnist.load_data()
#print(len(train_x), len(test_x)) # 60000, 10000
#print(test_y[:5])


test_scaled = test_x.reshape(-1,28,28,1) /  255.0  # 이미지 데이터 정규화
print(test_scaled.shape)

# import matplotlib.pyplot as plt
# print(test_x[0].shape) # (28, 28)
# # 수치가 0 에 가까운것은  검정색으로
# # 수치가 255에 가까운 것은 흰색으로 표현
# plt.imshow(test_x[0] , cmap='gray')

# plt.savefig('figures/test_0.jpeg')


from tensorflow.keras.models import load_model
import joblib
import numpy as np

# 소숫점 이하 3자리까지 출력
np.set_printoptions(precision=8, suppress=True)

fsmodel = load_model('models/fashinmnist_model.keras')
fsmodel.summary()

classes = ['티셔츠', '바지', '스웨터', '드레스','코트', '샌달', '셔츠', '스니커즈', '가방', '앵클부츠']
# preds 가장 큰 인덱스 찾아리스트인덱스색인
classclf = np.array(classes)
pred = fsmodel.predict(test_scaled[:5])
print(classclf[np.argmax(pred, axis=1)])