import numpy as np
import pandas as pd

# e 지수 표현하는 과학적표기 대신  소수점 이하 8자리까지 표현
np.set_printoptions(precision=8, suppress=True)

fishdf = pd.read_csv('/home/sckit/deeplearning_prj/20260609/fish_data.csv')
print(fishdf)
print(fishdf['Species'].unique()) # 물고기의 종이 몇종? ( 총 7종의 물고기 )
# 7개 물고기 중 어떤 물고기야 ?  ( 다중분류 )
fishdf.info()

# Series 를 넘파이 배열로 변환해주는 메서드 ==> to_numpy()
fish_target = fishdf['Species'].to_numpy()
print(fish_target)
print(fishdf.columns)
fish_train = fishdf[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_train)


# train/test  데이터 분리
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = \
    train_test_split(fish_train, fish_target, random_state=42)

print(train_x.shape)
print(test_x.shape)
print(train_x[:5])
#  스케일 조정  => 표준점수정규화 ==> 스케일 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_x)
test_scaled = scaler.transform(test_x)
print(train_scaled[:5])

# 학습모델 준비
# sklearn 다중분류==> 로지스틱회귀 모델에서 특정 파라미터 설정만 해주면 됨
from sklearn.linear_model import LogisticRegression
muti_lrmodel = LogisticRegression(multi_class='multinomial', max_iter=1000, C=20)

# 학습(fit)
#print(train_y) #  00001
muti_lrmodel.fit(train_scaled, train_y)

# 모델 성능평가
print( muti_lrmodel.score(train_scaled, train_y)) # 0.932
print( muti_lrmodel.score(test_scaled, test_y) ) # 0.925 성능

# 예측
# 테스트 데이터중 첫번째 하나만 어떤 종인지 예측?
print( test_scaled[:3].shape )
pred = muti_lrmodel.predict(test_scaled[:3])
print(pred)

print( muti_lrmodel.predict_proba(test_scaled[:3]) )
print( muti_lrmodel.classes_ )


from tensorflow.keras.models import Sequential
