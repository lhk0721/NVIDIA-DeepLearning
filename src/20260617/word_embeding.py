import numpy as np
from tensorflow.keras.layers import Embedding

input_data = np.array(
    [
        [3,4,7],
        [9,2,3],
        [1,6,499]
    ]
)
# 2를 499로 바꾸고, 밀집벡터의 한 행이 16열이 되도록 하자.

embedding = Embedding(
    input_dim=500, # 안맞으면 00000으로 나온다.
    output_dim=16,
    input_length=3
)

output = embedding(input_data)
print(output)