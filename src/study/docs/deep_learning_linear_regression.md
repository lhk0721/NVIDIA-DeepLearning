# 딥러닝 선형회귀 (Keras)

> 관련 코드: [`deep_learning/linear_regression.py`](../deep_learning/linear_regression.py)

## 개념

신경망(Dense 층)을 쌓아 **연속값을 예측**하는 회귀 모델을 만든다.
예제는 농어의 길이로 무게를 예측한다 (캐글 Fish Market 데이터).

## 전처리

### 1) reshape — 입력을 2차원으로
scikit-learn / Keras는 입력을 `(샘플 수, 특성 수)` 2차원으로 받는다.

```python
train_x = train_x.reshape(-1, 1)   # (42,) → (42, 1)
```

### 2) 다항 특성(Polynomial Feature) 추가
무게는 길이에 **비선형**으로 증가하므로, 길이의 제곱 항을 특성으로 추가한다.
이렇게 하면 직선이 아닌 곡선 관계를 학습할 수 있다.

```python
train_poly = np.column_stack((train_x**2, train_x))   # [길이², 길이] → 특성 2개
```

## 모델 설계 — Sequential

`Sequential`은 층을 순서대로 쌓는 가장 기본적인 모델 구조다.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=4, input_dim=2, activation='leaky_relu'))  # 입력층 (특성 2개)
model.add(Dense(units=8, activation='leaky_relu'))               # 은닉층
model.add(Dense(units=1, activation='linear'))                   # 출력층 (회귀)
```

- `units`: 뉴런 개수
- `input_dim`: 입력 특성 수 (첫 층에만 지정)
- `activation`: 활성화 함수 → [활성화 함수 문서](./activation_functions.md) 참고
  - 은닉층: `leaky_relu` (비선형)
  - **회귀 출력층: `linear`** (값 범위 제한 없이 그대로 출력)
- `model.summary()`로 층 구조와 파라미터 수 확인

## 컴파일 (compile)

학습 방법을 설정한다.

```python
model.compile(
    loss='mse',          # 손실 함수: 평균제곱오차 (회귀의 기본)
    optimizer='adam',    # 옵티마이저: 가중치 갱신 방법
    metrics=['mae'],     # 평가 지표: 평균절대오차
)
```

| 항목 | 회귀에서 자주 쓰는 값 | 의미 |
|------|----------------------|------|
| loss | `mse` | 예측-정답 차이의 제곱 평균. 학습이 최소화하려는 대상 |
| optimizer | `adam` | 가장 무난한 기본 옵티마이저 |
| metrics | `mae` | 사람이 해석하기 쉬운 오차 (단위 그대로) |

## 학습 (fit)

```python
model.fit(
    train_poly, train_y,
    batch_size=1,    # 한 번에 학습할 샘플 수
    epochs=500,      # 전체 데이터 반복 학습 횟수
    verbose=1,       # 진행 로그 출력
)
```

- `epochs`: 데이터셋 전체를 몇 번 반복 학습할지
- `batch_size`: 가중치를 한 번 갱신할 때 사용하는 샘플 수 (작을수록 자주 갱신)

## 평가 / 예측

```python
model.evaluate(test_poly, test_y)   # [loss, mae] 반환
pred = model.predict(train_poly[:5])
```

## 분류 vs 회귀 — 출력층 비교

| 구분 | 출력층 활성화 | 손실 함수(loss) |
|------|---------------|-----------------|
| 회귀 | `linear` | `mse` |
| 이진 분류 | `sigmoid` | `binary_crossentropy` |
| 다중 분류 | `softmax` | `categorical_crossentropy` |
