import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------------------------------------
# [1] 하이퍼파라미터 및 경로 설정
# -------------------------------------------------------------
TEST_DIR = r'/home/dlgusrb/deeplearning_prg/dataset/intel image dataset/seg_test'

IMG_SIZE = (128, 128)
BATCH_SIZE = 64

# -------------------------------------------------------------
# [2] 테스트 데이터셋 읽어오기 + 전처리
# -------------------------------------------------------------
raw_test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None,
    shuffle=False
)

def preprocess_for_autoencoder(images):
    images = tf.cast(images, tf.float32) / 255.0
    return images, images

test_ds = (raw_test_ds
           .map(preprocess_for_autoencoder, num_parallel_calls=tf.data.AUTOTUNE)
           .prefetch(tf.data.AUTOTUNE))

print("데이터셋 준비 완료!")

# -------------------------------------------------------------
# [3] 모델 로드 및 예측
# -------------------------------------------------------------
model = load_model(r'/home/dlgusrb/deeplearning_prg/models/autoencoding_intelimg.keras')

# 테스트 이미지 한 배치 꺼내기
for input_images, _ in test_ds.take(1):
    test_images = input_images.numpy()      # (BATCH_SIZE, 128, 128, 3)
    break

# 예측 (sigmoid 출력이므로 [0,1] 범위, 안전하게 clip)
predict_img = np.clip(model.predict(test_images), 0.0, 1.0)

# -------------------------------------------------------------
# [4] 결과 시각화: 원본 vs 복원
# -------------------------------------------------------------
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    # 원본
    ax1 = plt.subplot(2, n, i + 1)
    ax1.imshow(test_images[i])
    ax1.axis('off')
    if i == 0:
        ax1.set_title('Original', loc='left')

    # 복원
    ax2 = plt.subplot(2, n, i + 1 + n)
    ax2.imshow(predict_img[i])
    ax2.axis('off')
    if i == 0:
        ax2.set_title('Reconstructed', loc='left')

plt.tight_layout()
plt.savefig('figures/intel_autoencoder_result.png')
plt.close()
print("결과 저장 완료: figures/intel_autoencoder_result.png")
