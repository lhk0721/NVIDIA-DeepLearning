from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

## 1) 학습 때 저장한 모델 불러오기
newmodel = load_model(r'models/simplernn_imdb.keras')
# newmodel.summary()

## 데이터
new_review = 'Honestly? This movie is an absolute blast. Forget about boring dialogs and 30-minute character setups. You get a fight in the first 5 minutes, and the tournament kicks off by minute 15. This is exactly what we wanted from the 2021 movie. MK2 totally delivers. If you are a Mortal Kombat fan, this is pure joy to watch!'

new_review_onlyEng = re.sub(r'[^a-zA-Z\s]+','',new_review.lower())
new_review_onlyEng_clean = re.sub(r'[\s]+',' ',new_review_onlyEng)
# print(new_review_onlyEng_clean,type(new_review_onlyEng_clean))

data = re.findall(r'[a-zA-Z]+',new_review_onlyEng)
# print(data)
# ['honestly', 'this', 'movie', 'is', 'an', 'absolute', 'blast', 'forget', 'about', 'boring', 'dialogs', 'and', 'minute', 'character', 'setups', 'you','get', 'a', 'fight', 'in', 'the', 'first', 'minutes', 'and', 'the', 'tournament', 'kicks', 'off', 'by', 'minute', 'this', 'is', 'exactly', 'what', 'we', 'wanted', 'from', 'the', 'movie', 'mk', 'totally', 'delivers', 'if', 'you', 'are', 'a', 'mortal', 'kombat', 'fan', 'this', 'is', 'pure', 'joy', 'to', 'watch']

##

word_index = imdb.get_word_index()
# print(word_index.items())
# ...('geysers', 52003), ('artbox', 88582), ('cronyn', 52004), ('hardboiled', 52005), ("voorhees'", 88583), ('35mm', 16815), ("'l'", 88584), ('paget', 18509), ('expands', 20597)])

# print(word_index.keys())


# 0<PAD> (패딩)
# 1<START> (시퀀스 시작)
# 2<UNK> (unknown — num_words를 벗어나거나 oov_char로 대체된 단어)
# 3<UNUSED>
# 4~실제 단어 (빈도 1위 단어가 4번)

data_dict = {v:i for i,v in enumerate(data)}
# print(data_dict)


for word,_ in data_dict.items():
    if word in word_index.keys():
        data_dict[word] = word_index[word]+3
    else:
        data_dict[word] = 2

# print(data_dict)
# {'honestly': 1249, 'this': 11, 'movie': 17, 'is': 6, 'an': 32, 'absolute': 1554, 'blast': 5151, 'forget': 856, 'about': 41, 'boring': 354, 'dialogs':3230, 'and': 2, 'minute': 783, 'character': 106, 'setups': 19960, 'you': 22, 'get': 76, 'a': 3, 'fight': 545, 'in': 8, 'the': 1, 'first': 83, 'minutes': 231, 'tournament': 10822, 'kicks': 3412, 'off': 122, 'by': 31, 'exactly': 615, 'what': 48, 'we': 72, 'wanted': 470, 'from': 36, 'mk': 36030, 'totally': 481, 'delivers': 1542, 'if': 45, 'are': 23, 'mortal': 6942, 'kombat': 32659, 'fan': 334, 'pure': 1047, 'joy': 1802, 'to': 5, 'watch': 103}

data_num = [0]

for word, index in data_dict.items():
    if int(index) > 500:
        data_num.append(2)
    else:
        data_num.append(int(index))


# print(data_num)
# [0, 2, 11, 17, 6, 32, 2, 2, 2, 41, 354, 2, 2, 2, 106, 2, 22, 76, 3, 2, 8, 1, 83, 231, 2, 2, 122, 31, 2, 48, 72, 470, 36, 2, 481, 2, 45, 23, 2, 2, 334, 2, 2, 5, 103]

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

data_seq = pad_sequences(
    [data_num],
    maxlen = 100
) 

# print(len(data_seq[0]))

##
pred = newmodel.predict(data_seq)
print(pred)