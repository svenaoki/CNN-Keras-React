from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
# (W - F *2P)/S + 1
# l1 (3,6,5) + pool
# (128 - 5)/1 + 1 = 6x124x124
# (124 - 2)/2 + 1 = 6x62x62

# l2 (6,16,3) + pool
# (62 - 3) + 1 = 16x60x60
# (60 - 2)/2 + 1 = 16x30x30

# l3 (16,6,3) + pool
# (29 - 3) + 1 = 6x27x27
# (27 - 2)/2 + 1 = 6x14x14


def convNet():
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(5, 5),
                     activation="relu", input_shape=(128, 128, 3)))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(120, activation="relu"))
    model.add(Dense(84, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))
    return model
