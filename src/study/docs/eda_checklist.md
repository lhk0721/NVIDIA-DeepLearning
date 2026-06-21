# EDA (탐색적 데이터 분석) 체크리스트

> 관련 코드(사례):
> - KNN: [`../../20260604/knnClassifier_fish.py`](../../20260604/knnClassifier_fish.py), [`../../20260604/knn_standardScailed.py`](../../20260604/knn_standardScailed.py)
> - SVM: [`../machine_learning/svm/svm_classification.py`](../machine_learning/svm/svm_classification.py)
> - 결정트리: [`../../20260608/DecisionTree_모델분류.py`](../../20260608/DecisionTree_모델분류.py)

## EDA란

**EDA (Exploratory Data Analysis, 탐색적 데이터 분석)** 는 모델을 만들기 전에
데이터를 **요약·시각화해서 어떤 데이터인지 파악하는 단계**다.

핵심은 "구경"이 아니라 **다음 행동을 정하기 위해서** 본다는 것이다.
EDA의 모든 출력은 "그래서 무슨 결정을 내릴까?"라는 질문으로 이어져야 한다.

> 머신러닝 절차에서 EDA는 보통 **2번(전처리) 직전·도중**에 한다.
> 1. 데이터셋 선택 → **EDA / 전처리** → 3. train/test 분리 → 4. 모델 준비 →
> 5. 학습(fit) → 6. 평가(score) → 7. 예측(predict)

## 무엇을 확인하고, 무슨 결정을 내리나 (일반 체크리스트)

| # | 확인 항목 | 보는 도구 | 내릴 결정 |
|---|-----------|-----------|-----------|
| 1 | **결측치** (빈 값) | `info()`, `isnull().sum()` | 있으면 채우기(impute)/삭제, 없으면 정제 생략 |
| 2 | **자료형 (dtype)** | `info()`, `dtypes` | 문자열 라벨 → 인코딩, 숫자 → 그대로 |
| 3 | **식별자 vs 특성** | `head()` | 이름/ID 컬럼은 특성 아님 → `drop` |
| 4 | **값의 범위·스케일** | `describe()`, `head()` | 특성 간 범위 차 크고 거리기반 모델이면 → 스케일링 |
| 5 | **클래스 균형** | `value_counts()` | 불균형이면 stratify·class_weight·리샘플링, accuracy 신뢰 주의 |
| 6 | **train/test 일관성** | 양쪽 `info()`·`value_counts()` | 특성 컬럼·라벨 집합이 같은지 확인 |
| 7 | **이상치 (outlier)** | `describe()`, 산점도, `boxplot` | 튀는 값 확인·처리 여부 결정 |
| 8 | **특성 간 관계** | 산점도, `corr()` | 분리 가능성·중복 특성 파악 |

## 주요 도구 한눈에

```python
df.shape                  # (행 수, 열 수)
df.info()                 # 컬럼별 dtype + 결측치(non-null) 요약
df.head() / df.sample(5)  # 실제 값 몇 줄 보기
df.describe()             # 수치 컬럼 통계(평균/표준편차/min/max/사분위)
df.isnull().sum()         # 컬럼별 결측치 개수
df['label'].value_counts()# 범주별 개수 (클래스 균형 확인)
df.corr()                 # 수치 특성 간 상관계수
plt.scatter(...) / hist   # 분포·관계 시각화
```

---

## 사례 1 — KNN 생선 분류 (불균형 + 스케일 차이)

> [`knnClassifier_fish.py`](../../20260604/knnClassifier_fish.py),
> [`knn_standardScailed.py`](../../20260604/knn_standardScailed.py)

특성: 길이(length), 무게(weight) / 타깃: 도미(1) 35마리, 빙어(0) 14마리

**EDA에서 보인 것 → 내린 결정:**

| 관찰 | 결정 |
|------|------|
| 도미 35 : 빙어 14 → **클래스 불균형** | `train_test_split(..., stratify=fish_target)` 로 분할 시 비율 유지 |
| 길이 ≈ 10~40, 무게 ≈ 6~1000 → **스케일 차이 큼** | KNN은 **거리 기반**이라 큰 값(무게)이 거리를 지배 → **표준화 필수** |
| (스케일링) | 평균·표준편차로 표준점수 변환, **train의 mean/std로 test·예측도 동일 변환** |

> 핵심 교훈: **거리 기반 모델(KNN, SVM)** 은 특성 스케일이 다르면 한 특성이 거리를
> 독점한다. EDA에서 범위 차를 발견하면 스케일링을 결정해야 한다.
> 또한 학습·테스트·예측은 **같은 기준(train의 통계)** 으로 변환해야 한다.

## 사례 2 — SVM iris 분류 (균형 데이터)

> [`svm_classification.py`](../machine_learning/svm/svm_classification.py)

특성: petal_length, petal_width / 타깃: setosa·versicolor·virginica (각 50개)

**EDA에서 보인 것 → 내린 결정:**

| 관찰 | 결정 |
|------|------|
| 세 클래스 각 50개 → **균형** | accuracy로 평가해도 공정, 불균형 대응 불필요 |
| 두 특성 모두 cm 단위, 결측 없음 | 정제 불필요, 바로 학습 |
| 꽃잎(petal) 두 특성만으로 클래스가 잘 갈림 | 2개 특성만 선택 → 결정경계 **2D 시각화** 가능 |

## 사례 3 — 결정트리 서울 권역 분류 (식별자 + 균형 + 비스케일 모델)

> [`DecisionTree_모델분류.py`](../../20260608/DecisionTree_모델분류.py)

train: 자치구 20개 좌표 / test: 동 20개 좌표 / 타깃: 4개 권역

**EDA에서 보인 것 → 내린 결정:**

| 관찰 (`info`/`head`/`value_counts`) | 결정 |
|------|------|
| 모든 컬럼 20 non-null → **결측 0** | 정제 단계 생략 |
| `district`/`dong`은 object(문자열)이고 train·test 컬럼명도 다름 | **특성 아님 → `drop`** (식별자일 뿐) |
| `label`이 object(문자열) | `LabelEncoder`로 숫자화 |
| 특성 = 위도·경도(숫자), 범위는 다름(37 vs 127) | **결정트리는 스케일 영향 없음** → 그대로 사용 (KNN·SVM이면 스케일링 필요) |
| 4권역 각 5개로 **5:5:5:5 완전 균형** | accuracy로 평가 공정, 불균형 대응 불필요 |
| train·test의 **라벨 집합 동일** | train/test 호환 ✅, `le.transform`도 안전 |

> 핵심 교훈: **모델에 따라 같은 EDA 관찰의 결론이 달라진다.**
> "특성 범위가 다르다"는 사실은 KNN·SVM이면 "스케일링하라"는 신호지만,
> 트리 계열이면 "신경 쓸 것 없다"가 된다. EDA는 **쓸 모델까지 염두에 두고** 해석한다.

## 한 줄 요약

EDA는 **데이터를 보고 전처리·모델·평가 방법을 결정하는 단계**다.
출력을 볼 때마다 항상 묻자: **"그래서 무슨 결정을 내려야 하지?"**

> 출력별 즉시 판단표는 [pandas_basics.md](./pandas_basics.md#eda-출력--판단-체크리스트)에도 정리돼 있다.
