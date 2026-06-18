import cv2  # opencv-python
import numpy as np

# 소숫점 이하 3자리까지 출력
np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력


img = cv2.imread('src/20260612/sandal_1.jpg', cv2.IMREAD_GRAYSCALE)
print(img)
print(img.shape) # (330, 330) ==> (28,28)
# 원이미지 사이즈를 학습한 모델의 입력 사이즈로 변경(맞춰주어야함)
# cv2.INTER_AREA ==> 이 보간법을 사용
img_resize = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_AREA)
print(img_resize.shape)

# import matplotlib.pyplot as plt
# plt.imshow(img_resize , cmap='gray')

# plt.savefig('figures/sandal_resize.jpeg')

# 이미지 색상 반전
img_reverted = cv2.bitwise_not(img_resize)
print(img_reverted)

# import matplotlib.pyplot as plt
# plt.imshow(img_reverted , cmap='gray')
# plt.savefig('figures/sandal_reverted.jpeg')

img_reverted = img_reverted / 255.0  # 스케일 변환
print(img_reverted.shape)
img_reverted = img_reverted.reshape(1,28,28,1)
print(img_reverted.shape)

from tensorflow.keras.models import load_model

# 소숫점 이하 8자리까지 출력
np.set_printoptions(precision=8, suppress=True)
fsmodel = load_model('models/fashinmnist_model.keras')
fsmodel.summary()

# 위에서 생성한 임의의 데이터를 예측
pred = fsmodel.predict(img_reverted)
print(pred)

classes = ['티셔츠', '바지', '스웨터', '드레스','코트', '샌달', '셔츠', '스니커즈', '가방', '앵클부츠']
# preds 가장 큰 인덱스 찾아리스트인덱스색인
classclf = np.array(classes)
print(classclf[np.argmax(pred, axis=1)])