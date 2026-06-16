
train_dir = r'/home/dlgusrb/deeplearning_prg/dataset/cats_and_dogs/training_set'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_image_generator = ImageDataGenerator(
    rescale = 1.0/255.,
    rotation_range = 20,
    height_shift_range = 2.0
)

# image를 읽어 들이면서 이미지를 증강시켜주는 generator 생성
train_data_gen = train_image_generator.flow_from_directory(
    train_dir, # 불러올 이미지 경로
    batch_size = 2,
    shuffle = False,
    save_to_dir = r'/home/dlgusrb/deeplearning_prg/dataset/cats_and_dogs/temp',
    save_prefix = 'gen',
    save_format = 'jpg',
    target_size = (150, 150) # cnn모델 입력 사이즈로 리사이즈해라
)

i = 0
for b in train_data_gen:
    i += 1
    if i > 2:
        break
