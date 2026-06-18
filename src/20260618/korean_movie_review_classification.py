import numpy as np
import pandas as pd

pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)

# 한국어 불용어 제거 데이터 로딩
train_df = pd.read_csv(r'/home/dlgusrb/deeplearning_prg/dataset/korean_movie_rating/train_stopwords_reviews.csv', index_col=0)
test_df = pd.read_csv(r'/home/dlgusrb/deeplearning_prg/dataset/korean_movie_rating/test_stopwords_reviews.csv',index_col=0)
print(train_df.head(10))
print(test_df.head(10))
print(train_df.info())
print(test_df.info())

# 한국어 불용어 제거로 발생한 결측 데이터 삭제
print(train_df.loc[train_df['document'].isnull()])
print(test_df.loc[test_df['document'].isnull()])
train_df.dropna(how='any',inplace=True)
test_df.dropna(how='any',inplace=True)


# 훈련, 테스트 데이터 토큰화 및 정수 인코딩
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_size = 11775 # 단어 빈도수 체크 결과에 따른 11775개 단어 집합 사용
tokenizer = Tokenizer(word_size) # ( 0 예약 패딩, 1~11774까지 단어 집합 사용 )
tokenizer.fit_on_texts(train_df['document'])
print(tokenizer.word_index) # 21800 개 이상의 단어가 존재

# train 데이터와 test data 리뷰 문장을 0 패딩, 1~11774까지 단어 집합 사용해 정수 시퀀스 데이터 형태로 변환
# 변환된 데이터를 'sequences'컬럼으로 추가
train_df['sequences'] = tokenizer.texts_to_sequences(train_df['document'])
test_df['sequences'] = tokenizer.texts_to_sequences(test_df['document'])

train_df.reset_index(drop=True, inplace=True)  # 인덱스 초기화
test_df.reset_index(drop=True, inplace=True)   # 인덱스 초기화

print(train_df[25:30])
print(test_df[57:62])

# 11775 개 단어 집합만 고려 했음으로 빈도수가 1 이하인 단어로 이루어진 문장은 텅빈( [ ] )
# 형태로 변환 됨, 따라서 해당 문장의 인덱스를 찾아 제거 해줌
drop_train_idx = [idx for idx, sentence in enumerate(train_df['sequences']) if len(sentence) < 1]
print('drop_train_idx : \n', drop_train_idx)

drop_test_idx = [idx for idx, sentence in enumerate(test_df['sequences']) if len(sentence) < 1]
print('drop_test_idx : \n', drop_test_idx)

# 텅빈([ ]) sequence 데이터 위치 인덱스 활용해서  Dataframe 해당 행 삭제
train_df.drop(drop_train_idx,axis=0, inplace=True)
test_df.drop(drop_test_idx, axis=0, inplace=True)

train_df.reset_index(drop=True, inplace=True)  # 인덱스 초기화
test_df.reset_index(drop=True, inplace=True)   # 인덱스 초기화

print("========= 삭제 완료 검증 수행 ===========")
for idx, sequence in enumerate(train_df['sequences']):
    if(len(sequence) < 1):
        print(idx, sequence)

print(train_df[25:30])
print(test_df[57:62])

# 타깃 라벨 추출
y_train = np.array(train_df['label'])
y_test = np.array(test_df['label'])

print(len(train_df['sequences']))   # 최종 훈련데이터 31901 개 샘플
print(len(y_train))                     # 최종 훈련데이터 라벨 31901 개
print(len(test_df['sequences']))    # 최종 테스트데이터 31554 개 샘플
print(len(y_test))                      # 최종 테스트데이터 라벨 31554 개

train_review_sequences_len = [len(sequence) for sequence in  train_df['sequences']]
train_review_sequences_arr = np.array(train_review_sequences_len)
print('max : ', np.max(train_review_sequences_arr))  # 훈련 리뷰데이터 최대 길이 63
print('mean : ', np.mean(train_review_sequences_arr)) # 평균 길이 10.734114918027648

# # import matplotlib.pyplot as plt
# # plt.hist(train_review_sequences_len, bins=50)
# # plt.show() # pad 적용 30 길이로 동일하게 맞추자
#
X_train_pades = pad_sequences(train_df['sequences'], maxlen=30)
X_test_pades = pad_sequences(test_df['sequences'], maxlen=30)

print(len(X_train_pades[0]))
print(X_train_pades[:1])
print(len(X_test_pades[0]))
print(X_test_pades[:1])

# 최종 LSTM 모델 훈련 데이터 준비 완료

## LSTM 모델 준비
# X_train_pades, y_train
# X_test_pades, y_test
from tensorflow.keras.layers import Embedding, Dense, LSTM, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

embedding_dim = 100 # embedding 밀집벡터 차원
hidden_units = 256 # LSTM 뉴런수

model = Sequential()

optimizer = Adam(
    learning_rate = 1e-5
)

model.add(Input(shape = (30,)))

model.add(Embedding(
    word_size,
    embedding_dim
))

model.add(LSTM( hidden_units ))

model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer,
    metrics = ['accuracy']
)

EarlyStopCB = EarlyStopping(
    monitor='val_loss', 
    verbose=1, 
    patience=4, 
    restore_best_weights=True
)

ModelCheckCB = ModelCheckpoint(
    'models/korean_movie_review_best_model.h5', 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True
)

# 모델 훈련
history = model.fit(
    X_train_pades,
    y_train,
    epochs=50,
    validation_data = (X_test_pades, y_test),
    callbacks=[EarlyStopCB, ModelCheckCB],
    batch_size=64
)

## ============ 예측 결과 검증 ============
# restore_best_weights=True 라서 model 은 현재 best 가중치를 보유

# 1) 테스트셋 전체 손실/정확도
test_loss, test_acc = model.evaluate(X_test_pades, y_test, verbose=0)
print('\n========= 테스트셋 평가 =========')
print('테스트 손실(loss)     : {:.4f}'.format(test_loss))
print('테스트 정확도(accuracy): {:.4f} ({:.2f}%)'.format(test_acc, test_acc * 100))

# 2) 예측 확률 -> 0/1 라벨 변환 (임계값 0.5)
y_prob = model.predict(X_test_pades).flatten()  # 시그모이드 확률
y_pred = (y_prob > 0.5).astype(int)             # 0:부정, 1:긍정

# 3) 혼동행렬 / 정밀도·재현율·F1
from sklearn.metrics import confusion_matrix, classification_report

print('\n========= 혼동행렬 (행:실제, 열:예측) =========')
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(
    cm,
    index=['실제_부정(0)', '실제_긍정(1)'],
    columns=['예측_부정(0)', '예측_긍정(1)']
))

print('\n========= 분류 리포트 =========')
print(classification_report(
    y_test, y_pred,
    target_names=['부정(0)', '긍정(1)'],
    digits=4
))

# 4) 실제 예측이 어떻게 나오는지 샘플 10개 직접 확인
print('\n========= 샘플 예측 비교 (테스트셋 앞 10개) =========')
sample_df = test_df.reset_index(drop=True)
for i in range(10):
    real = '긍정' if y_test[i] == 1 else '부정'
    pred = '긍정' if y_pred[i] == 1 else '부정'
    mark = 'O' if y_test[i] == y_pred[i] else 'X'
    print('[{}] 정답:{} | 예측:{} (확률 {:.2f}%) | {}'.format(
        mark, real, pred, y_prob[i] * 100, sample_df['document'].iloc[i][:30]
    ))

