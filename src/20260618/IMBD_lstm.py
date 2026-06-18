from tensorflow.keras.datasets import imdb
import numpy as np
from sklearn.model_selection import train_test_split

( train_x, train_y ), ( test_x , test_y )=\
    imdb.load_data(
        num_words = 500 # 최대 빈도수 단어 500개만 활용
    )

# print(len(train_x))
# print(len(test_x))
# 25000
# 25000

# print(train_y)
# [1 0 0 ... 0 1 0]
# 라벨도 수치형으로 되어있다.

# print(np.unique(train_y, return_counts=True))
# (array([0, 1]), array([12500, 12500])) # 긍정, 부정이 절반씩 쪼개져있다.
# 0: 부정, 1: 긍정

(train_x, test_x, train_y, test_y) = \
    train_test_split(
        train_x, train_y,
        test_size=0.2,
        random_state= 43
    )

# print(len(train_x), len(test_x))
# 20000 5000


## 길이가 다른 리뷰 정수데이터 배열을 길이가 동잃한 정수 배열로 변경한다.
# print(len(train_x[0]))
# print(len(train_x[1]))
# 138
# 154 # 길이가 다르다. 하지만 타임스탬프 맞춰줘야 하므로, 적절한 길이를 길이 분포 상 적절한 값으로 일관 변경해줘야 한다.
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 길이를 모두 100으로 맞추는데, 짧은건 2로 채우고, 긴거는 버린다!
train_seq = pad_sequences(
    train_x,
    maxlen = 100
)

test_seq = pad_sequences(
    test_x,
    maxlen = 100
)

# print(len(train_seq[0]))
# 100


## 정수 벡터를 그대로 투입하면 가중치가 너무 적다. 열 해쉬벡터로 펼쳐넣어줘야 한다.
# one_hot encoding으로 하면, 500개 배열로 해야한다... 메모리 overflow 나기 쉽다..
# 대안이 밀집벡터. 워드 임베딩 한 단어 스칼라를 해시 가능한 실수로 표현
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Embedding, LSTM, Dense

# embedding = Embedding(
#     input_dim=501,
#     output_dim = 16,
#     input_length=100
# )

# train_emb = embedding(train_seq)
# test_emb = embedding(test_seq)

# print(train_emb, train_emb.shape)

embedding_dim = 16 # embedding 밀집벡터 차원
hidden_units = 8 # LSTM 뉴런수

rnnmodel = Sequential()
rnnmodel.add(
    Input(shape=(100,))
)
rnnmodel.add(
    Embedding(input_dim=500, output_dim=embedding_dim, input_length=100)
)
rnnmodel.add(
    SimpleRNN(hidden_units)
    # - val_accuracy: 0.7892 - val_loss: 0.4669
    # LSTM(hidden_units)
    # - val_accuracy: 0.7982 - val_loss: 0.4358
)

rnnmodel.add(
    Dense(1, activation='sigmoid')
)
rnnmodel.summary()

# print(train_emb[0])

## 모델 컴파일
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


checkpoint_cb = ModelCheckpoint(
    filepath='models/simplernn_imdb.keras',
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)

earlystop_cv = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

optimizer = Adam(
    learning_rate = 1e-4
)

rnnmodel.compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer,
    metrics = ['accuracy']
)

history = rnnmodel.fit(
    train_seq,
    train_y,
    validation_data = (test_seq, test_y),
    verbose = 1,
    epochs = 100,
    batch_size = 64,
    callbacks = [checkpoint_cb, earlystop_cv]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt

epochs = np.arange(len(acc))

plt.figure()
plt.plot(
    epochs,
    loss,
    'b',
    label = 'Train_Loss'
)

plt.plot(
    epochs,
    val_loss,
    'orange',
    label = 'Val_Loss'
)

plt.legend()
plt.savefig('figures/imbd_simple_rnn.jpeg')

##
# print(train_x[0]) # 이미 정수 벡터화 되어있다.
# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5,...

# wordIndex = imdb.get_word_index() # 어떤 단어를 어떤 수치로 변환했는가
# # print(wordIndex)

# # for word, idx in wordIndex.items():
# #     if idx == 1:
# #         print(word,idx)
# #     if idx == 2:
# #         print(word,idx)
# #     if idx == 3:
# #         print(word,idx)
# #     if idx == 4:
# #         print(word,idx)


# conv_word_index =dict([ (idx+3, word) for (word, idx) in wordIndex.items() ])

# # for word, index in conv_word_index.items():

# decode_centence = ' '.join([
#     conv_word_index[i] if i in conv_word_index.keys() else '?'  for i in train_x[0]
# ])

# print(decode_centence)

##

