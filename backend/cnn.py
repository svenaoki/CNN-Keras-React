
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from utils import convNet

# setting paths and some initial parameter
print(os.getcwd())
PATH = os.path.join(os.getcwd(), 'backend',)
TRAIN_DIR = os.path.join(PATH, "dataset", "train")
TEST_DIR = os.path.join(PATH, "dataset", "test")
CLASSES = 'Cat', 'Dog'
BATCH_SIZE = 6

# loading data and transformations
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_loader = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(128, 128), batch_size=BATCH_SIZE, class_mode='binary')
test_loader = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(128, 128),  batch_size=BATCH_SIZE, class_mode='binary')

# get one example
for i, (input, label) in enumerate(train_loader):
    for idx in range(BATCH_SIZE):
        plt.subplot(2, 3, idx+1)
        plt.imshow(input[idx])
        plt.title(CLASSES[np.int(label[idx])])
    # plt.show()
    if i == 0:
        break

# hyperparamters
NUM_EPOCHS = 20
LEARNING_RATE = 0.01

# initialize model, optimizer and loss criterion
model = convNet()
print(model.summary())
model.compile(optimizer=optimizers.Adam(),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# training loop
history = model.fit(
    train_loader,
    steps_per_epoch=len(train_loader)//BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=test_loader,
    validation_steps=len(test_loader)//BATCH_SIZE)

# save model
model.save_weights(os.path.join(PATH, "model.h5"))

val_acc = history.history["val_accuracy"]
acc = history.history["accuracy"]
epochs = range(1, NUM_EPOCHS)

plt.plot(epochs, val_acc, label="Validation", color="b")
plt.plot(epochs, acc, label="Training", color="r")
