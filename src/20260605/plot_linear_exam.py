import numpy as np
import matplotlib.pyplot as plt

## [5,5]와 [50,50] 위치를 직선으로 연결해서 표시
# [[x축인덱스],[y축인덱스]]

# x = np.linspace(-5, 5, 200)      # -5~5 구간을 200개 점으로
# y = 2*x**3 - 3*x**2 + x - 5      # 예: 2x³ - 3x² + x - 5

# plt.plot(x, y)
# plt.grid(True)
# plt.savefig('figures/line.jpeg')

# plt.plot([[10,20,30],[5,20,60]])
# plt.savefig('figures/line.jpeg')

##

# 머신러닝 모델에서의 회귀선
# x = (LRmodel.coef_*15 + LRmodel.intercept_)
# y = (LRmodel.coef_*15 + LRmodel.intercept_)

# plt.plot([
#     [15, 50],
#     []
# ])

##