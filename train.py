import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame

import config

# init Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# limit the action-space to
#   0. walk right
#   1. jump right
#   2. jump
env = JoypadSpace(
    env,
    [
        ['right'],
        ['right', 'A'],
        ['A']
    ]
)

# apply wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)
env.reset()

# init the save directory
save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None # eg. Path('checkpoints/2021-08-01T18-25-27/mario.chkpt')
print("[INFO] State dimensions:", (4, 84, 84))
print("[INFO] Action dimensions:", env.action_space.n)

# init the Mario agent
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

# init the logger
logger = MetricLogger(save_dir)
# init number of episodes to train
episodes = config.EPISODES

### train the model EPISODE no. of times by playing the game
for e in range(episodes):

    state = env.reset()

    while True:

        # Show environment
        # env.render()

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Agent caches experience to memory
        mario.store_to_memory(state, next_state, action, reward, done)

        # Agent learns 
        q, loss = mario.learn()

        # Logging is performed
        logger.log_step(reward, loss, q)

        # Update agent's state
        state = next_state

        # Check if game is done
        if done or info['flag_get']:
            break

    logger.log_episode()

    # for every 20 episodes, log the metrics 
    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
