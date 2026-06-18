
from tensorflow.keras.models import load_model # 학습된 모델을 불러올때 사용


titanic_bestmodel = load_model('models/titanic_bestmodel.keras')
titanic_bestmodel.summary()

# 머신러닝 처럼  새로운 3사람의 정보를  만들어서
# 생존여부 예측 / 추론
