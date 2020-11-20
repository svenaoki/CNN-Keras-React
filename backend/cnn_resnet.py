
import math
import os

PATH = os.path.join(os.getcwd(), 'backend', 'python')
TRAIN_DIR = os.path.join(PATH, "dataset", "train")
TEST_DIR = os.path.join(PATH, "dataset", "test")
CLASSES = 'Cat', 'Dog'
BATCH_SIZE = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparamters
NUM_EPOCHS = 2
ITER_PER_EPOCH = math.ceil(len(train_set)/BATCH_SIZE)
LEARNING_RATE = 0.01
PATH_CHECKPOINT = os.path.join(PATH, "checkpoint_dict_model.pt")
PATH_MODEL = os.path.join(PATH, "state_dict_model.pt")

# load resnet model


n_correct = 0
n_samples = 0


# save model

# save checkpoint

n_correct = 0
n_samples = 0
