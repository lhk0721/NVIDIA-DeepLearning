## 시그모이드 함수
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)
print(x)

y = 1 / ( 1 + np.exp(-x) )
print(y)

# plt.plot(x,y)
# plt.savefig('figures/sigmoid.jpeg')
