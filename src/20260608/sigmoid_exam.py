import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)
print(x)
y = 1 / ( 1 + np.exp(-x) )  # 시그모이드(sigmoid) 함수
print(y)

plt.plot(x,y)
plt.savefig('sigmoid.jpeg')