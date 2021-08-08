import torch
from collections import deque

MEMORY = deque(maxlen=1000)
BATCH_SIZE = 32

EXPLORATION_RATE = 1
EXPLORATION_RATE_DECAY = 0.999975
EXPLORATION_RATE_MIN = 0.1
GAMMA = 0.9

CURR_STEP = 0
BURNIN = 33  # min. experiences before training
LEARN_EVERY = 400  # no. of experiences between updates to Q_online
SYNC_EVERY = 100   # no. of experiences between Q_target & Q_online sync

SAVE_EVERY = 5e5   # no. of experiences between saving Mario Net

OPTIMIZER_LEARNING_RATE = 0.00025
LOSS_FN = torch.nn.SmoothL1Loss()

EPISODES = 500