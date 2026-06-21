# 전처리기의 fit / transform 원리

> 관련 코드:
> - LabelEncoder: [`../../20260608/DecisionTree_모델분류.py`](../../20260608/DecisionTree_모델분류.py)
> - 스케일링: [`../../20260604/knn_standardScailed.py`](../../20260604/knn_standardScailed.py)

## 한 줄 요약

전처리기의 `fit`은 **모델을 훈련**하는 게 아니라 **"변환 규칙을 데이터에서 만들어 기억"**하는 것이다.
train에서 만든 그 규칙을 test·예측에 `transform`으로 **동일하게** 적용해야 데이터가 어긋나지 않는다.

## fit / transform / fit_transform

| 메서드 | 하는 일 |
|--------|---------|
| `fit(데이터)` | 데이터를 스캔해 **변환 규칙(기준)을 생성·기억** |
| `transform(데이터)` | 기억한 규칙대로 **실제 변환** |
| `fit_transform` | 위 둘을 한 번에 (보통 **train에만** 사용) |

> `fit`은 "학습"이라 부르지만 모델 학습과 다르다. **데이터를 봐야 규칙을 정할 수 있어서**
> 데이터를 한 번 훑는 것뿐이다.

## 예 1 — LabelEncoder (문자열 → 숫자)

sklearn 모델은 `'Gangseo'` 같은 문자열을 못 받는다. 숫자로 바꿔야 하는데,
**어떤 문자열을 어떤 숫자로 바꿀지 대응표**가 필요하다. 그 표를 만드는 게 `fit`이다.

`fit`이 하는 일:
1. 고유 라벨 수집 → `{'Gangbuk','Gangdong','Gangnam','Gangseo'}`
2. **알파벳순 정렬** → `['Gangbuk','Gangdong','Gangnam','Gangseo']`
3. 순서대로 번호 부여 → `Gangbuk=0, Gangdong=1, Gangnam=2, Gangseo=3`

```python
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

y_encoded      = le.fit_transform(train_y.values.ravel())  # train: 대응표 생성 + 변환
le.classes_                                                # ['Gangbuk' 'Gangdong' 'Gangnam' 'Gangseo']
                                                           #  인덱스(0,1,2,3)가 곧 부여된 번호
test_y_encoded = le.transform(test_y.values.ravel())       # test: 같은 표로 변환만
```

## 예 2 — 스케일링 (표준화)

`fit`이 기억하는 규칙이 여기선 **평균(mean)·표준편차(std)** 다.

```python
mean = np.mean(train_x, 0)     # train에서 기준(평균) 계산  ← fit에 해당
std  = np.std(train_x, 0)      # train에서 기준(표준편차) 계산

train_scaled = (train_x - mean) / std    # train 변환
test_scaled  = (test_x  - mean) / std    # test도 "train의" mean/std로 변환  ← transform에 해당
new_scaled   = ([25,150] - mean) / std   # 예측 데이터도 동일 기준
```

> sklearn `StandardScaler`를 쓰면 `scaler.fit_transform(train)` / `scaler.transform(test)` 로
> 위 과정을 자동화한다. mean/std를 내부에 기억하는 구조는 동일하다.

## 진짜 핵심 — 왜 train만 fit하고 test는 transform만?

**train과 test가 같은 변환 규칙을 써야 하기 때문이다.**

만약 test에 `fit_transform`을 따로 하면 규칙이 train과 달라질 수 있다.

| 전처리기 | test를 따로 fit하면 생기는 문제 |
|----------|--------------------------------|
| LabelEncoder | test에 일부 클래스가 빠지거나 순서가 다르면 같은 `'Gangnam'`이 train=2, test=1로 **번호가 어긋남** → 정확도 엉터리 |
| Scaler | test의 평균·표준편차로 정규화하면 train과 **다른 좌표계**가 됨 → 모델이 배운 기준과 불일치 |

> `knn_standardScailed.py` 주석: **"학습, 예측 모두 같은 mean, std로 정규화해야 한다. scale 맞추기!!"**

즉 변환 기준은 **항상 train에서 한 번만 정하고(fit)**, test·예측은 거기에 맞춘다(transform).
이것은 데이터 누수(data leakage)를 막는 원칙이기도 하다 — test 정보가 변환 규칙에 새어들면 안 된다.

## 올바른 패턴 vs 흔한 실수

```python
# ✅ 올바름 — train에서만 fit
enc.fit_transform(train)
enc.transform(test)
enc.transform(new_data)

# ❌ 실수 — test를 따로 fit (규칙이 달라짐 / 데이터 누수)
enc.fit_transform(train)
enc.fit_transform(test)
```

## 정리

- 전처리기의 `fit` = **변환 규칙을 데이터에서 만들어 기억** (LabelEncoder=대응표, Scaler=mean/std)
- `transform` = 그 규칙대로 변환
- **train에서만 `fit`**, test·예측은 같은 규칙으로 `transform`만 → 기준 일관성 유지
- 이 원리는 모든 sklearn 전처리기에 공통이다 (`LabelEncoder`, `StandardScaler`, `OneHotEncoder` 등)
