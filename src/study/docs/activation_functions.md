# 활성화 함수 (Activation Function)

> 관련 코드: [`activation/sigmoid.py`](../activation/sigmoid.py)

## 개념

활성화 함수는 뉴런의 입력(가중치 합)을 **비선형**으로 변환해 출력한다.
비선형 변환이 없으면 층을 아무리 쌓아도 결국 하나의 선형식이 되므로,
신경망이 복잡한 패턴을 학습하려면 반드시 비선형 활성화 함수가 필요하다.

## 시그모이드 (Sigmoid)

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

```python
import numpy as np

x = np.arange(-5, 5, 0.1)
y = 1 / (1 + np.exp(-x))
```

### 특징
- 출력 범위가 **0 ~ 1** → 확률처럼 해석 가능
- **이진 분류**의 출력층 활성화 함수로 사용 (출력이 0.5 이상이면 양성)
- S자(시그모이드) 곡선 형태

### 한계
- **기울기 소실(Vanishing Gradient)**: 입력이 매우 크거나 작으면 기울기가 0에 가까워져
  깊은 신경망에서 학습이 잘 안 된다. → 은닉층에는 ReLU 계열을 주로 사용.

## 함께 보면 좋은 활성화 함수

| 함수 | 출력 범위 | 주 용도 |
|------|-----------|---------|
| Sigmoid | 0 ~ 1 | 이진 분류 출력층 |
| Softmax | 0 ~ 1 (합=1) | 다중 분류 출력층 |
| ReLU | 0 ~ ∞ | 은닉층 (기본값) |
| Leaky ReLU | -∞ ~ ∞ | 은닉층 (ReLU의 죽은 뉴런 보완) |
| Linear | -∞ ~ ∞ | 회귀 출력층 |

> Leaky ReLU / Linear의 실제 사용 예는 [딥러닝 선형회귀](./deep_learning_linear_regression.md) 참고.
