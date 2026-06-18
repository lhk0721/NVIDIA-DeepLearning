import numpy as np
import pandas as pd
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 50)
pd.set_option('display.colheader_justify','center')  # 컬럼 중앙 출력


##
review_train = pd.read_csv(r'/home/dlgusrb/deeplearning_prg/dataset/korean_movie_rating/ratings_train.csv',header=0,delimiter='\t',quoting=3)
review_test = pd.read_csv(r'/home/dlgusrb/deeplearning_prg/dataset/korean_movie_rating/ratings_test.csv',header=0,delimiter='\t',quoting=3)

# print(review_train)
# print(review_train.info())
#  #   Column    Non-Null Count  Dtype  
# ---  ------    --------------  -----  
#  0   id        32652 non-null  int64  
#  1   document  32651 non-null  object 
#  2   label     32651 non-null  float64
# dtypes: float64(1), int64(1), object(1)
# memory usage: 765.4+ KB
# None

##
review_train.dropna(how='any', inplace=True)
# review_test.dropna(how='any', inplace=True)

# print(review_train.info())
#  #   Column    Non-Null Count  Dtype  
# ---  ------    --------------  -----  
#  0   id        32650 non-null  int64  
#  1   document  32650 non-null  object 
#  2   label     32650 non-null  float64
# dtypes: float64(1), int64(1), object(1)
# memory usage: 1020.3+ KB
# None

# print(review_train.head())
#          id                                           document  label
# 0   9976970                                아 더빙.. 진짜 짜증나네요 목소리    0.0
# 1   3819312                  흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나    1.0
# 2  10265843                                  너무재밓었다그래서보는것을추천한다    0.0
# 3   9045019                      교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정    0.0
# 4   6483659  사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 ...    1.0

##
# review_train['label'] = review_train['label'].map(lambda x: int(x))
review_train['label'] = review_train['label'].astype('Int64')
# review_test['label'] = review_test['label'].map(lambda x: int(x))

# print(review_train.head())

##
# print(review_train['document'].nunique())
# 32163

review_train.drop_duplicates(subset='document',inplace=True)
# review_test.drop_duplicates(subset='document',inplace=True)

# print(review_train.info())
#  #   Column    Non-Null Count  Dtype 
# ---  ------    --------------  ----- 
#  0   id        32163 non-null  int64 
#  1   document  32163 non-null  object
#  2   label     32163 non-null  Int64 
# dtypes: Int64(1), int64(1), object(1)

##
import re
import numpy as np

def OnlyKorean(arg):
    return re.sub(r'[^ㄱ-힣\s]','',arg)

review_train['document'] = review_train['document'].map(OnlyKorean)

def DropMultiSpacing(arg):
    return re.sub(r'[\s]+',' ',arg)

review_train['document'] = review_train['document'].map(DropMultiSpacing)

review_train.info()

# print(review_train['document'])

##
from konlpy.tag import Okt
from tqdm import tqdm #처리 상태를 막대 bar로 표시
import os

okt = Okt()

# 불용어(stopwords): 감성과 무관한 조사·어미·일반어. 토큰화 후 제거에 사용.
stopwords = [
    '의', '가', '이', '은', '들', '는', '좀', '잘', '걸', '와', '과',
    '도', '를', '으로', '자', '에', '한', '하다', '고', '을', '인',
    '듯', '네', '지', '임', '게', '만', '겠', '에서', '에게', '까지',
    '부터', '이나', '거나', '면서', '그', '저', '것', '수', '때', '거',
    '그리고', '그러나', '하지만', '그래서', '그런데', '또', '및',
    '더', '아주', '매우', '정말', '진짜', '너무', '왜', '어떤',
    '무슨', '이런', '저런', '그런', '이렇게', '저렇게', '그렇게',
]

X_train = []
for sentence in tqdm(review_train['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 각 문장을 토큰화
    sentence_removed_stopwords = \
    [word for word in tokenized_sentence if not word in stopwords] # 불용어제거#불용어 제거된 단어 리스트를 한 문장으로 합친 다음 X_train list 에 추가
    X_train.append(' '.join(sentence_removed_stopwords))

print(review_train[:5])
print('=*80')
print(X_train[:5])

review_train['document'] = X_train