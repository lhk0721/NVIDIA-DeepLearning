import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score # 정확도평가
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

dataSet = pd.read_csv(r'dataset/spam/spam.csv')
# print(dataSet.head())
#   Category                                            Message
# 0      ham  Go until jurong point, crazy.. Available only ...
# 1      ham                      Ok lar... Joking wif u oni...
# 2     spam  Free entry in 2 a wkly comp to win FA Cup fina...
# 3      ham  U dun say so early hor... U c already then say...
# 4      ham  Nah I don't think he goes to usf, he lives aro...

dataSet_100 = dataSet.iloc[:].copy()
# print(dataSet_100)
#    Category                                            Message
# 0       ham  Go until jurong point, crazy.. Available only ...
# 1       ham                      Ok lar... Joking wif u oni...
# 2      spam  Free entry in 2 a wkly comp to win FA Cup fina...
# 3       ham  U dun say so early hor... U c already then say...
# 4       ham  Nah I don't think he goes to usf, he lives aro...
# ..      ...                                                ...
# 95     spam  Your free ringtone is waiting to be collected....
# 96      ham                  Watching telugu movie..wat abt u?
# 97      ham  i see. When we finish we have loads of loans t...
# 98      ham  Hi. Wk been ok - on hols now! Yes on for a bit...
# 99      ham                    I see a cup of coffee animation

dataSet_100['Category'] = dataSet_100['Category'].map({'ham':0,'spam':1})
# print(dataSet_100)
#     Category                                            Message
# 0          1  Go until jurong point, crazy.. Available only ...
# 1          1                      Ok lar... Joking wif u oni...
# 2          0  Free entry in 2 a wkly comp to win FA Cup fina...
# 3          1  U dun say so early hor... U c already then say...
# 4          1  Nah I don't think he goes to usf, he lives aro...
# ..       ...                                                ...
# 95         0  Your free ringtone is waiting to be collected....
# 96         1                  Watching telugu movie..wat abt u?               
# 97         1  i see. When we finish we have loads of loans t...
# 98         1  Hi. Wk been ok - on hols now! Yes on for a bit...
# 99         1                    I see a cup of coffee animation

def clean_message(arg):
    text = arg.lower()
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    text = re.sub(r'\s+',' ',text)
    return text.strip()

dataSet_100['Message'] = dataSet_100['Message'].map(clean_message)

# print(dataSet_100['Message'].head())
# 0    go until jurong point crazy available only in ...
# 1                                ok lar joking wif oni
# 2    free entry in wkly comp to win fa cup final tk...
# 3                dun say so early hor already then say
# 4    nah dont think he goes to usf he lives around ...
# Name: Message, dtype: object

##
np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력

cv = CountVectorizer(binary=True)
train_x = cv.fit_transform(dataSet_100['Message'])
train_x_encoded = train_x.toarray()
# print(train_x)

train_y = dataSet_100['Category'].astype('int').copy()

##

bnb = BernoulliNB()
bnb.fit(train_x_encoded,train_y)
# print(len(cv.get_feature_names_out()))

##

temp = [
    "WINNER!! You have been selected to receive a 900 prize reward! To claim call 09061701461 now",  # spam
    "URGENT! Your mobile number has won 2000 pounds in the weekly draw. Txt CLAIM to 80086 to collect",  # spam
    "FREE entry into our wkly comp to win the latest camera phone, just text WIN to 83600",  # spam
    "Had your mobile 11 months or more? U R entitled to update to the latest colour mobiles for FREE",  # spam
    "Congrats! 1 year special cashback voucher is awaiting you. Call 08712300220 to claim before it expires",  # spam
    "Are we still on for lunch tomorrow at half twelve?",  # ham
    "Ur ringtone service has been changed. Reply STOP to cancel or call 09064019788 for help",  # spam
    "Hiya, will be home a bit late tonight, can u pick up some milk on the way back",  # ham
    "PRIVATE! Your account statement shows 800 unredeemed points. Call 08719899217 to claim",  # spam
    "Sorry I missed your call earlier, give me a ring when you get a chance",  # ham
]

temp_mail_cv = cv.transform(temp).toarray()
# print(temp_mail_cv)
print(bnb.predict(temp_mail_cv))  # 0=ham, 1=spam
# [1 1 1 1 1 0 1 0 1 0]