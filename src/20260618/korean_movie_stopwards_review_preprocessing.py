import numpy as np
import pandas as pd
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 50)
pd.set_option('display.colheader_justify','center')  # 컬럼 중앙 출력


##
train_df = pd.read_csv(r'/home/dlgusrb/deeplearning_prg/dataset/korean_movie_rating/train_stopwords_reviews.csv', index_col=0)
test_df = pd.read_csv(r'/home/dlgusrb/deeplearning_prg/dataset/korean_movie_rating/test_stopwords_reviews.csv',index_col=0)

# train_df.info()
#  #   Column              Non-Null Count  Dtype 
# ---  ------              --------------  ----- 
#  0   ,id,document,label  31968 non-null  object
# dtypes: object(1)
# memory usage: 249.9+ KB
# <class 'pandas.core.frame.DataFrame'>

# test_df.info()
#  #   Column              Non-Null Count  Dtype 
# ---  ------              --------------  ----- 
#  0   ,id,document,label  31663 non-null  object
# dtypes: object(1)
# memory usage: 247.5+ KB


##
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# train_df.info()
# test_df.info()

# print(train_df.head())
#       id                         document                       label
# 0   9976970                                   아 더빙 진짜 짜증나다 목소리    0  
# 1   3819312                   흠 포스터 보고 초딩 영화 줄 오버 연기 조차 가볍다 않다    1  
# 2  10265843                              너 무재 밓었 다그 래서 보다 추천 다    0  
# 3   9045019                        교도소 이야기 구먼 솔직하다 재미 없다 평점 조정    0  
# 4   6483659  사이 몬페 그 익살스럽다 연기 돋보이다 영화 스파이더맨 에서 늙다 보이다 커스틴 던...    1  

##
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_size = 11775 # imbd의 num_words의 역할

tokenizer = Tokenizer(word_size)

tokenizer.fit_on_texts(train_df['document'])
# print(tokenizer.word_index)

# for word, index in tokenizer.word_index.items():
#     if(index == 2):
#         print(word)

train_df['sequences'] = tokenizer.texts_to_sequences(train_df['document'])
test_df['sequences'] = tokenizer.texts_to_sequences(test_df['document'])
print(train_df.head())
print(test_df.head())

##
from tensorflow.keras.preprocessing import pad_sequences

