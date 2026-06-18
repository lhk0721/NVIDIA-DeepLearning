# 새로운 리뷰 데이터 예측
from konlpy.tag import Okt
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os
import re

pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)

# 한국어 불용어 제거 데이터 로딩
train_df = pd.read_csv(r'/home/dlgusrb/deeplearning_prg/dataset/korean_movie_rating/train_stopwords_reviews.csv', index_col=0)
test_df = pd.read_csv(r'/home/dlgusrb/deeplearning_prg/dataset/korean_movie_rating/test_stopwords_reviews.csv',index_col=0)

# 한국어 불용어 제거로 발생한 결측 데이터 삭제
train_df.dropna(how='any',inplace=True)
test_df.dropna(how='any',inplace=True)


# 훈련, 테스트 데이터 토큰화 및 정수 인코딩
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_size = 11775 # 단어 빈도수 체크 결과에 따른 11775개 단어 집합 사용
tokenizer = Tokenizer(word_size) # ( 0 예약 패딩, 1~11774까지 단어 집합 사용 )
tokenizer.fit_on_texts(train_df['document'])

best_model = load_model('models/korean_movie_review_best_model.h5')

okt = Okt()

##

stopwords = [
    '의', '가', '이', '은', '들', '는', '좀', '잘', '걸', '와', '과',
    '도', '를', '으로', '자', '에', '한', '하다', '고', '을', '인',
    '듯', '네', '지', '임', '게', '만', '겠', '에서', '에게', '까지',
    '부터', '이나', '거나', '면서', '그', '저', '것', '수', '때', '거',
    '그리고', '그러나', '하지만', '그래서', '그런데', '또', '및',
    '더', '아주', '매우', '정말', '진짜', '너무', '왜', '어떤',
    '무슨', '이런', '저런', '그런', '이렇게', '저렇게', '그렇게',
]

def new_review_predict(review_string): 
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]','', review_string)
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어제거print(new_sentence) # ['영화', '굿', '잼']
    # [new_sentence] : 불용어 처리된 단어 리스트를 정수 인코딩 sequences 데이터 형성을# 위해 하나로 묶어서([ ]) 변환해 줘야함
    encoded = tokenizer.texts_to_sequences( [new_sentence] ) # 정수 인코딩
    print(encoded) # [[1, 363, 334]] 
    sentence_padding = pad_sequences(encoded, maxlen = 30) # 패딩 적용 동일 길이 Sequences 형성print(sentence_padding)
    #[[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 1 363 334]] 
    score = float(best_model.predict(sentence_padding) ) # new_sentence 예측
    if(score > 0.5): 
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100)) 
    else: 
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))


new_review_predict('이 영화 굿 잼')
new_review_predict('이렇게 재미없는 영화는 처음')
new_review_predict('뭐 이런 영화가 다 있어')
new_review_predict('에잇 돈 날렸네')
new_review_predict('이 영화 꼭 추천 도장 꽉!')


## ============ 저장된 모델의 테스트셋 정확도 검증 ============
import numpy as np

# 테스트 리뷰를 학습 때와 동일하게 정수 인코딩
test_eval = test_df.copy()
test_eval['sequences'] = tokenizer.texts_to_sequences(test_eval['document'])

# 단어집합(11775개)에 없어서 빈 시퀀스가 된 행은 학습 때처럼 제거
test_eval = test_eval[test_eval['sequences'].map(len) > 0]

X_test = pad_sequences(test_eval['sequences'], maxlen=30)   # 길이 30 패딩
y_test = test_eval['label'].astype(int).to_numpy()          # 0:부정, 1:긍정

# 1) 전체 손실/정확도
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print('\n========= 테스트셋 평가 (저장된 best_model) =========')
print('테스트 샘플 수        : {}'.format(len(y_test)))
print('테스트 손실(loss)     : {:.4f}'.format(test_loss))
print('테스트 정확도(accuracy): {:.4f} ({:.2f}%)'.format(test_acc, test_acc * 100))

# 2) 예측 확률 -> 0/1 변환 후 혼동행렬 / 분류 리포트
y_prob = best_model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)

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

# [[1, 367, 338]]
# I0000 00:00:1781774544.908145  127105 cuda_dnn.cc:461] Loaded cuDNN version 92300
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 690ms/step
# /home/dlgusrb/deeplearning_prg/src/20260618/korean_movie_review_classification_predict.py:64: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
#   score = float(best_model.predict(sentence_padding) ) # new_sentence 예측
# 86.47% 확률로 긍정 리뷰입니다.

# [[64, 1, 97]]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
# 94.20% 확률로 부정 리뷰입니다.

# [[63, 42, 1, 9, 6]]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
# 71.47% 확률로 부정 리뷰입니다.

# [[7500, 96, 1749]]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
# 80.05% 확률로 부정 리뷰입니다.

# [[1, 144, 222, 5171]]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
# 83.35% 확률로 긍정 리뷰입니다.