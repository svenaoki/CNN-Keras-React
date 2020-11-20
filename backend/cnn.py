import os
import math

import matplotlib.pyplot as plt

# setting paths and some initial parameter
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
PATH = os.path.join(os.getcwd(), 'backend', 'python')
TRAIN_DIR = os.path.join(PATH, "dataset", "train")
TEST_DIR = os.path.join(PATH, "dataset", "test")
CLASSES = 'Cat', 'Dog'
BATCH_SIZE = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transformations

# loading data


# hyperparamters
NUM_EPOCHS = 2
ITER_PER_EPOCH = math.ceil(len(train_set)/BATCH_SIZE)
LEARNING_RATE = 0.01
PATH_CHECKPOINT = os.path.join(PATH, "checkpoint_dict_model.pt")
PATH_MODEL = os.path.join(PATH, "state_dict_model.pt")

# initialize model, optimizer and loss criterion


# training loop
n_correct = 0
n_samples = 0


# save model

# save checkpoint

# load model for validation set


# test loop
n_correct = 0
n_samples = 0
