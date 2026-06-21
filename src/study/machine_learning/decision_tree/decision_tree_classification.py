import pandas as pd

district_dict_list = [
    {'district': 'Gangseo-gu', 'latitude': 37.551000, 'longitude': 126.849500, 'label': 'Gangseo'},
    {'district': 'Yangcheon-gu', 'latitude': 37.52424, 'longitude': 126.855396, 'label': 'Gangseo'},
    {'district': 'Guro-gu', 'latitude': 37.4954, 'longitude': 126.8874, 'label': 'Gangseo'},
    {'district': 'Geumcheon-gu', 'latitude': 37.4519, 'longitude': 126.9020, 'label': 'Gangseo'},
    {'district': 'Mapo-gu', 'latitude': 37.560229, 'longitude': 126.908728, 'label': 'Gangseo'},

    {'district': 'Gwanak-gu', 'latitude': 37.487517, 'longitude': 126.915065, 'label': 'Gangnam'},
    {'district': 'Dongjak-gu', 'latitude': 37.5124, 'longitude': 126.9393, 'label': 'Gangnam'},
    {'district': 'Seocho-gu', 'latitude': 37.4837, 'longitude': 127.0324, 'label': 'Gangnam'},
    {'district': 'Gangnam-gu', 'latitude': 37.5172, 'longitude': 127.0473, 'label': 'Gangnam'},
    {'district': 'Songpa-gu', 'latitude': 37.503510, 'longitude': 127.117898, 'label': 'Gangnam'},

    {'district': 'Yongsan-gu', 'latitude': 37.532561, 'longitude': 127.008605, 'label': 'Gangbuk'},
    {'district': 'Jongro-gu', 'latitude': 37.5730, 'longitude': 126.9794, 'label': 'Gangbuk'},
    {'district': 'Seongbuk-gu', 'latitude': 37.603979, 'longitude': 127.056344, 'label': 'Gangbuk'},
    {'district': 'Nowon-gu', 'latitude': 37.6542, 'longitude': 127.0568, 'label': 'Gangbuk'},
    {'district': 'Dobong-gu', 'latitude': 37.6688, 'longitude': 127.0471, 'label': 'Gangbuk'},

    {'district': 'Seongdong-gu', 'latitude': 37.557340, 'longitude': 127.041667, 'label': 'Gangdong'},
    {'district': 'Dongdaemun-gu', 'latitude': 37.575759, 'longitude': 127.025288, 'label': 'Gangdong'},
    {'district': 'Gwangjin-gu', 'latitude': 37.557562, 'longitude': 127.083467, 'label': 'Gangdong'},
    {'district': 'Gangdong-gu', 'latitude': 37.554194, 'longitude': 127.151405, 'label': 'Gangdong'},
    {'district': 'Jungrang-gu', 'latitude': 37.593684, 'longitude': 127.090384, 'label': 'Gangdong'}
]

dong_dict_list = [
    {'dong': 'Gaebong-dong', 'latitude': 37.489853, 'longitude': 126.854547, 'label': 'Gangseo'},
    {'dong': 'Gochuk-dong', 'latitude': 37.501394, 'longitude': 126.859245, 'label': 'Gangseo'},
    {'dong': 'Hwagok-dong', 'latitude': 37.537759, 'longitude': 126.847951, 'label': 'Gangseo'},
    {'dong': 'Banghwa-dong', 'latitude': 37.575817, 'longitude': 126.815719, 'label': 'Gangseo'},
    {'dong': 'Sangam-dong', 'latitude': 37.577039, 'longitude': 126.891620, 'label': 'Gangseo'},

    {'dong': 'Nonhyun-dong', 'latitude': 37.508838, 'longitude': 127.030720, 'label': 'Gangnam'},
    {'dong': 'Daechi-dong', 'latitude': 37.501163, 'longitude': 127.057193, 'label': 'Gangnam'},
    {'dong': 'Seocho-dong', 'latitude': 37.486401, 'longitude': 127.018281, 'label': 'Gangnam'},
    {'dong': 'Bangbae-dong', 'latitude': 37.483279, 'longitude': 126.988194, 'label': 'Gangnam'},
    {'dong': 'Dogok-dong', 'latitude': 37.492896, 'longitude': 127.043159, 'label': 'Gangnam'},

    {'dong': 'Pyoungchang-dong', 'latitude': 37.612129, 'longitude': 126.975724, 'label': 'Gangbuk'},
    {'dong': 'Sungbuk-dong', 'latitude': 37.597916, 'longitude': 126.998067, 'label': 'Gangbuk'},
    {'dong': 'Ssangmoon-dong', 'latitude': 37.648094, 'longitude': 127.030421, 'label': 'Gangbuk'},
    {'dong': 'Ui-dong', 'latitude': 37.648446, 'longitude': 127.011396, 'label': 'Gangbuk'},
    {'dong': 'Samcheong-dong', 'latitude': 37.591109, 'longitude': 126.980488, 'label': 'Gangbuk'},

    {'dong': 'Hwayang-dong', 'latitude': 37.544234, 'longitude': 127.071648, 'label': 'Gangdong'},
    {'dong': 'Gui-dong', 'latitude': 37.543757, 'longitude': 127.086803, 'label': 'Gangdong'},
    {'dong': 'Neung-dong', 'latitude': 37.553102, 'longitude': 127.080248, 'label': 'Gangdong'},
    {'dong': 'Amsa-dong', 'latitude': 37.552370, 'longitude': 127.127124, 'label': 'Gangdong'},
    {'dong': 'Chunho-dong', 'latitude': 37.547436, 'longitude': 127.137382, 'label': 'Gangdong'}
]

train_df = pd.DataFrame(district_dict_list)
test_df = pd.DataFrame(dong_dict_list)

## 데이터 전처리

## train_df.info()
#  #   Column     Non-Null Count  Dtype  
# ---  ------     --------------  -----  
#  0   district   20 non-null     object 
#  1   latitude   20 non-null     float64
#  2   longitude  20 non-null     float64
#  3   label      20 non-null     object 

## test_df.info()
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   dong       20 non-null     object
#  1   latitude   20 non-null     float64
#  2   longitude  20 non-null     float64
#  3   label      20 non-null     object 

## print(train_df.head())
#        district   latitude   longitude    label
# 0    Gangseo-gu  37.551000  126.849500  Gangseo
# 1  Yangcheon-gu  37.524240  126.855396  Gangseo
# 2       Guro-gu  37.495400  126.887400  Gangseo
# 3  Geumcheon-gu  37.451900  126.902000  Gangseo
# 4       Mapo-gu  37.560229  126.908728  Gangseo

## print(test_df.head())
#            dong   latitude   longitude    label
# 0  Gaebong-dong  37.489853  126.854547  Gangseo
# 1   Gochuk-dong  37.501394  126.859245  Gangseo
# 2   Hwagok-dong  37.537759  126.847951  Gangseo
# 3  Banghwa-dong  37.575817  126.815719  Gangseo
# 4   Sangam-dong  37.577039  126.891620  Gangseo

## print(train_df['label'].value_counts())
# label
# Gangseo     5
# Gangnam     5
# Gangbuk     5
# Gangdong    5

## print(test_df['label'].value_counts())
# label
# Gangseo     5
# Gangnam     5
# Gangbuk     5
# Gangdong    5

## train, test 데이터 분할
train_x = train_df[['longitude','latitude']]
train_y = train_df[['label']]

test_x = test_df[['longitude','latitude']]
test_y = test_df[['label']]

## 모델 준비
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(train_y.values.ravel())
test_y_encoded = le.transform(test_y.values.ravel())

# print(train_y.values.ravel())
# ['Gangseo' 'Gangseo' 'Gangseo' 'Gangseo' 'Gangseo' 'Gangnam' 'Gangnam'
#  'Gangnam' 'Gangnam' 'Gangnam' 'Gangbuk' 'Gangbuk' 'Gangbuk' 'Gangbuk'
#  'Gangbuk' 'Gangdong' 'Gangdong' 'Gangdong' 'Gangdong' 'Gangdong']

# print(y_encoded)
# [3 3 3 3 3 2 2 2 2 2 0 0 0 0 0 1 1 1 1 1]

# print(le.classes_)
# ['Gangbuk' 'Gangdong' 'Gangnam' 'Gangseo']

## 모델 설계

model_dt = tree.DecisionTreeClassifier()

## 모델 학습
clf = model_dt.fit(
    train_x,
    y_encoded
)

## 성능평가

# print('acc: ', clf.score(test_x, test_y_encoded))
# acc:  0.95

## 결정 경계 시각화

def display_decision_surface(clf,x,y):

    x_min = x['longitude'].min() - 0.01
    x_max = x['longitude'].max() + 0.01

    y_min = x['latitude'].min() - 0.01
    y_max = x['latitude'].max() + 0.01

    xx, yy =\
        np.meshgrid(
            np.arange(x_min, x_max, 0.001),
            np.arange(y_min, y_max, 0.001)
        )
    
    np.set_printoptions(threshold=np.inf)

    z = clf.predict(
        np.column_stack(
            [xx.ravel(), yy.ravel()]
        )
    )

    z = z.reshape(xx.shape)

    plt.figure(figsize=(10,7))
    plt.contourf(
        xx,
        yy,
        z,
        cmap = plt.cm.RdYlBu
    )

    #"모델이 처음 보는 동네를 어느 권역으로 분류하는지" 확인하려고 넣은 새 예측 지점
    plt.scatter(
        126.9367,
        37.5562, 
        c = 'blue',
        marker = '^'
    )

    plt.title('Decision Surface of predict data', fontsize=16)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('figures/new_dtmode.jpeg')



display_decision_surface(clf, train_x, y_encoded)