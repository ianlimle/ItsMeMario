import random, datetime
from pathlib import Path
import argparse

import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame

import config

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint", type=str, required=True,
	help="path to checkpoint for a trained Mario agent")
args = vars(ap.parse_args())


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

checkpoint = Path(args["checkpoint"])
print("[INFO] State dimensions:", (4, 84, 84))
print("[INFO] Action dimensions:", env.action_space.n)

# init the Mario agent
mario = Mario(state_dim=(4, 84, 84), 
            action_dim=env.action_space.n, 
            save_dir=save_dir,
            memory=config.MEMORY, 
            exploration_rate_decay=config.EXPLORATION_RATE_DECAY,
            learn_every=config.LEARN_EVERY,
            sync_every=config.SYNC_EVERY,
            checkpoint=checkpoint)

mario.exploration_rate = mario.exploration_rate_min
# init the logger
logger = MetricLogger(save_dir)
# init number of episodes to train
episodes = config.EPISODES

for e in range(episodes):

    state = env.reset()

    while True:

        # Show environment
        env.render()

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Agent caches experience to memory
        mario.store_to_memory(state, next_state, action, reward, done)

        # Logging is performed
        logger.log_step(reward, None, None)

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
