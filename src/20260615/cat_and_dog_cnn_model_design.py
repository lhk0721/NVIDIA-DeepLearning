from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(
    Conv2D(
        16, 
        kernel_size=(3, 3), 
        input_shape=(150, 150, 3), 
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=2
    )
)

model.add(
    Conv2D(
        32, 
        (3, 3), 
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=2
    )
)

model.add(
    Conv2D(
        64, 
        (3, 3), 
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=2
    )
)

model.add(
    Flatten()
)
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()