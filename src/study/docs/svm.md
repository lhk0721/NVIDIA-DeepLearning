# SVM (Support Vector Machine, 서포트 벡터 머신)

> 관련 코드:
> - [`machine_learning/svm/svm_classification.py`](../machine_learning/svm/svm_classification.py) — 기본 분류
> - [`machine_learning/svm/svm_gridsearch.py`](../machine_learning/svm/svm_gridsearch.py) — 하이퍼파라미터 튜닝
> - [`machine_learning/svm/svm_visualization.py`](../machine_learning/svm/svm_visualization.py) — 결정경계 시각화

## 개념

SVM은 두 클래스를 나누는 **결정경계(decision boundary)** 를 찾되,
경계와 가장 가까운 데이터들(**서포트 벡터**)과의 거리(**마진, margin**)가
**최대**가 되도록 경계를 정하는 분류 알고리즘이다. 마진이 클수록 일반화 성능이 좋다.

## 커널 (Kernel)

직선으로 나눌 수 없는 데이터를 **고차원으로 매핑**해 비선형 경계를 그을 수 있게 해준다.

- `linear`: 선형 경계
- `rbf` (Radial Basis Function): 가장 많이 쓰는 비선형 커널. 곡선 경계를 만든다.

## 핵심 하이퍼파라미터

| 파라미터 | 의미 | 크게 하면 | 작게 하면 |
|----------|------|-----------|-----------|
| `C` (cost) | 오분류 허용 정도(규제 강도) | 과적합 ↑ | 과소적합 ↑ |
| `gamma` | 한 데이터의 영향 범위 (rbf) | 과적합 ↑ | 과소적합 ↑ |

> `C`, `gamma`를 키우면 경계가 학습 데이터에 딱 맞춰져 **과적합**,
> 줄이면 경계가 단순해져 **과소적합** 경향.

## 기본 사용법

```python
from sklearn.svm import SVC

model_svc = SVC(C=0.7, kernel='rbf', gamma=0.7)
model_svc.fit(train_x, train_y)

pred = model_svc.predict([[4.7, 1.7]])
```

## 과적합 / 과소적합 진단

학습 정확도(train acc)와 테스트 정확도(test acc)를 비교한다.

| train acc | test acc | 진단 |
|-----------|----------|------|
| 높음 | 높음 | 좋은 모델 |
| 높음 | 낮음 | **과적합** (외우기만 함) |
| 낮음 | 낮음 | **과소적합** (학습 자체가 덜 됨) |

```python
train_acc = model_svc.score(train_x.values, train_y.values.ravel())
test_acc  = model_svc.score(test_x.values,  test_y.values.ravel())
```

## 하이퍼파라미터 튜닝 — GridSearchCV

최적의 `C`, `gamma`, `kernel` 조합을 **교차검증(cross validation)** 으로 자동 탐색한다.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

svc_params = [{
    'kernel': ['rbf'],
    'gamma':  [0.1, 0.3, 0.5, 0.7, 1.0],
    'C':      [0.3, 0.7, 1, 1.3, 1.5],
}]

clf = GridSearchCV(SVC(), svc_params, cv=10)   # cv = 폴드 수
clf.fit(x, y)

clf.best_params_     # 최적 파라미터  예) {'C': 0.3, 'gamma': 0.7, 'kernel': 'rbf'}
clf.best_score_      # 최적 교차검증 점수
clf.best_estimator_  # 최적 모델 (바로 predict 가능)
```

- `cv=n`: 데이터를 n등분해 (n-1)개로 학습, 1개로 검증을 n번 반복 → 평균 성능으로 평가.
- 모든 파라미터 조합을 전수 탐색하므로 조합이 많으면 느리다.

## 결정경계 시각화 (등고선)

2개 특성(petal_length, petal_width)으로 학습한 모델의 경계를 격자(meshgrid)로 그린다.

```python
# 1) 격자 좌표 생성
x1s = np.linspace(x_min, x_max, 100)
y1s = np.linspace(y_min, y_max, 100)
x1, y1 = np.meshgrid(x1s, y1s)           # 100 x 100 = 10,000개 점

# 2) 모든 격자점 예측
xy1 = np.column_stack([x1.ravel(), y1.ravel()])
Z = model_svc.predict(xy1).reshape(x1.shape)

# 3) 경계선/영역 그리기
plt.contour(x1, y1, Z, levels=[0.0, 1.0, 2.0], colors='red')  # 경계선
plt.contourf(x1, y1, Z, cmap=plt.cm.RdYlBu, alpha=0.3)        # 영역 색칠
```

핵심 흐름: **격자 좌표 만들기 → 각 점을 모델로 예측 → 예측값을 등고선으로 표현**.
이렇게 하면 모델이 공간을 어떻게 클래스별로 나눴는지 한눈에 볼 수 있다.
