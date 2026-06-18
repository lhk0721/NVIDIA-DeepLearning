from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import numpy as np


# x = np.array([
#     [5,6,7],
#     [4,5,6],
#     [6,7,8],
#     [1,2,3]
# ])

# y = np.array([8,7,9,4])


##

import numpy as np

starts = np.arange(1, 101).reshape(-1, 1)   # 1 ~ 1000
x = starts + np.array([0, 1, 2])              # (1000, 3): [n, n+1, n+2]
y = starts.flatten() + 3
print(x,y)

x = x.reshape(-1, 3, 1)   
y = y.reshape(-1, 1)


##

# rnnModel = Sequential()

# rnnModel.add(
#     SimpleRNN(
#         10,
#         return_sequences = False,
#         input_shape = (3,1)
#     )
# )

# rnnModel.add(Dense(1))

# rnnModel.summary()

# rnnModel.compile(
#     loss = 'mse',
#     optimizer = 'adam',
#     metrics = ['mse']
# )


##

# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.callbacks import EarlyStopping

# checkpoint_cb = ModelCheckpoint(
#     filepath='models/simplernn.keras',
#     monitor='val_loss',
#     verbose=1,
#     save_best_only=True
# )

# earlystop_cv = EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     restore_best_weights=True
# )

# rnnModel.fit(
#     x,y,
#     epochs = 250,
#     batch_size = 1,
#     callbacks = [checkpoint_cb, earlystop_cv],
#     verbose = 1
# )

##
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import vgg16

newmodel = load_model(r'models/simplernn.keras')
# newmodel.summary()

start = np.arange(1,11).reshape(-1,1)
newx = start + np.array([0, 1, 2])

pred = newmodel.predict(newx)
print(pred)