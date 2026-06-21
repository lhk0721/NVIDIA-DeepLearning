# Pandas 기초

> 관련 코드: SVM·결정트리 예제에서 `pd.DataFrame`을 데이터 준비에 사용한다.
> - [`../machine_learning/svm/svm_classification.py`](../machine_learning/svm/svm_classification.py)
> - [`../../20260608/DecisionTree_모델분류.py`](../../20260608/DecisionTree_모델분류.py)

## DataFrame이란

행(row) × 열(column)로 이뤄진 **2차원 표** 자료구조. 엑셀 시트 한 장이라고 보면 된다.
한 열(Series)은 같은 의미의 데이터 묶음, 한 행은 하나의 샘플(레코드)이다.

## `pd.DataFrame()` — 무엇을 받나

딕셔너리만 받는 게 아니라 **"행과 열로 펼칠 수 있는 거의 모든 2차원 구조"**를 받는다.
받는 데이터는 두 부류로 갈린다.

- **이름표가 들어있는 구조** (dict, list of dict) → 컬럼명을 알아서 가져감
- **값만 있는 구조** (리스트, NumPy 배열) → 컬럼명을 `columns=`로 따로 줘야 함

### 1) 딕셔너리들의 리스트 (`list of dict`) — 한 행 = 한 딕셔너리

```python
data = [
    {'district': 'Gangseo-gu', 'latitude': 37.55, 'label': 'Gangseo'},
    {'district': 'Gangnam-gu', 'latitude': 37.51, 'label': 'Gangnam'},
]
df = pd.DataFrame(data)
```

- 딕셔너리 하나가 **한 행**, key가 **컬럼명**이 된다.
- 행마다 key가 달라도 됨 → 없는 값은 `NaN`으로 채워진다.
- 결정트리 예제(`DecisionTree_모델분류.py`)가 이 방식.

### 2) 딕셔너리 하나 (`dict`) — 한 key = 한 열

```python
data = {
    'district': ['Gangseo-gu', 'Gangnam-gu'],
    'latitude': [37.55, 37.51],
}
df = pd.DataFrame(data)
```

- key가 **컬럼명**, value(리스트)가 **그 열 전체**가 된다.
- 1번과 정반대 방향: 1번은 **행 단위**, 2번은 **열 단위**로 데이터를 넣는다.

### 3) 2차원 리스트 / NumPy 배열 — 값만, 컬럼명은 따로

```python
values = [[37.55, 126.84], [37.51, 127.04]]
df = pd.DataFrame(values, columns=['latitude', 'longitude'])
```

- 컬럼명을 안 주면 `0, 1, 2...` 정수가 자동으로 붙는다.
- `np.column_stack(...)` 결과를 넣는 SVM/iris 코드가 이 방식:

```python
iris_df = pd.DataFrame(
    np.column_stack((dataset['data'], dataset['target'])),
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
)
```

### 4) 그 외 입력

| 입력 형태 | 결과 |
|-----------|------|
| `pd.DataFrame()` | 빈 데이터프레임 |
| Series 여러 개 / Series의 dict | 각 Series가 한 열 |
| 튜플들의 리스트 `[(1,2),(3,4)]` | 2차원 리스트와 동일 |
| 다른 DataFrame | 복사본 |

### 보조 인자

```python
pd.DataFrame(data,
    columns=['a', 'b'],     # 열 이름 (또는 열 순서 지정/선택)
    index=['행1', '행2'],    # 행 이름 (기본은 0,1,2...)
)
```

## 자주 쓰는 조작 (머신러닝 데이터 준비 기준)

### 열 선택
```python
df['latitude']                 # 한 열 → Series (1차원)
df[['latitude', 'longitude']]  # 여러 열 → DataFrame (2차원)
```

> 머신러닝 특성 데이터(X)는 보통 2차원이어야 하므로 `df[['col']]`(대괄호 2개)로 뽑는다.

### 열 순서 변경 / 선택
```python
df = df[['district', 'longitude', 'latitude', 'label']]   # 원하는 순서로 재배치
```

### 열 삭제
```python
df.drop(['district'], axis=1, inplace=True)   # axis=1 = 열 방향, inplace=원본 수정
```

### 열 이름 변경
```python
df.rename(columns={'target': 'label'})
```

### 값만 꺼내기 (NumPy 배열로)
```python
df.values            # 2차원 ndarray
df.values.ravel()    # 1차원으로 평탄화 (정답 y를 (n,1) → (n,)로 만들 때)
```

### 빠른 탐색
```python
df.head()                    # 앞 5행
df.sample(5)                 # 무작위 5행
df['label'].value_counts()   # 범주별 개수 세기
df.shape                     # (행 수, 열 수)
df.info()                    # 컬럼별 타입·결측치 요약
df.describe()                # 수치 컬럼 통계 요약
```

## EDA 출력 → 판단 체크리스트

`info()`·`head()`·`value_counts()` 같은 출력은 **보기 위해서가 아니라 다음 행동을 정하려고** 본다.
각 출력에서 무슨 결정을 내려야 하는지 정리한다. (개념 전체는 [eda_checklist.md](./eda_checklist.md) 참고)

### `info()` → 결측치 · 자료형 · 특성 적격성

```
 #   Column     Non-Null Count  Dtype
 0   district   20 non-null     object     ← 문자열(식별자)
 1   latitude   20 non-null     float64    ← 숫자(특성)
 2   longitude  20 non-null     float64    ← 숫자(특성)
 3   label      20 non-null     object     ← 문자열(정답)
```

| 본 것 | 판단 |
|-------|------|
| 모두 `20 non-null` (결측 0) | 결측 처리(impute/dropna) **불필요** |
| `latitude`/`longitude`가 float | 특성으로 바로 사용 가능 |
| `district`가 object이고 train·test 이름컬럼 다름(`district`/`dong`) | 특성 아님 → **`drop`** |
| `label`이 object(문자열) | sklearn에 넣으려면 **`LabelEncoder`로 숫자화** |

### `head()` → 구조 확인 · X/y 분리 설계 · 스케일 감지

| 본 것 | 판단 |
|-------|------|
| 한 행 = (식별자, 위도, 경도, 권역) | `X = [['longitude','latitude']]`, `y = ['label']` 설계 |
| 위도 ~37, 경도 ~127로 **범위 다름** | 트리 → 무관 / KNN·SVM → **스케일링 필요** 신호 |

### `value_counts()` → 클래스 균형 · 평가지표 · train/test 호환

```
train: Gangseo 5, Gangnam 5, Gangbuk 5, Gangdong 5
test:  Gangseo 5, Gangnam 5, Gangbuk 5, Gangdong 5
```

| 본 것 | 판단 |
|-------|------|
| 4클래스 5:5:5:5 **완전 균형** | **accuracy로 평가 공정**, 불균형 대응(stratify/리샘플링) 불필요 |
| train·test **라벨 집합 동일** | train/test 호환 ✅, `le.transform` 안전 (새 라벨 없음) |

> 반례: KNN 생선 데이터는 도미 35 : 빙어 14로 **불균형** → `value_counts()`에서 이를 보고
> `train_test_split(stratify=...)` 를 결정했다. 출력이 다르면 결정도 달라진다.

## 핵심 직관

`pd.DataFrame`은 결국 **2차원 표**를 만드는 도구다.
- **dict / list of dict** → 이름표(컬럼명)가 데이터에 같이 있으니 그대로 가져감
- **리스트 / 배열** → 값만 있으니 `columns=`로 이름을 붙여줘야 함

머신러닝에서는 보통 이 표를 만든 뒤, 특성 열들(X)과 정답 열(y)을 골라
`model.fit(X, y)`에 넘기는 흐름으로 쓴다.
