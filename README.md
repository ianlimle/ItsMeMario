## Prerequisites
1. Install [conda](https://www.anaconda.com/products/individual)
2. Install dependencies with `environment.yml`
    ```
    conda env create -f environment.yml
    ```
    Check the new environment *mario-env* is [created successfully](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

3. Activate *mario-env* environment
    ```
    conda activate mario-env
    ```

4.  If your shell is not properly configured to use `conda activate`, it may be there is a source line in your bash scripts that has to explicitly reference your conda installation path. You can reference your conda installation path       with the following: 
    ```
    CONDA_PREFIX=$(conda info --base)
    source $CONDA_PREFIX/etc/profile.d/conda.sh
    conda activate mario-env
    ```


## Running the application
To start the **training** process for Mario,
```
python train.py
```
This starts the *double Q-learning* and logs key training metrics to `checkpoints`. A copy of `MarioNet` and current exploration rate will be saved.

To **evaluate** a trained Mario,
```
python evaluate.py
```
This visualizes Mario playing the game in a window. Performance metrics will be logged to a new folder under `checkpoints`. Change the `load_dir`, e.g. `checkpoints/2021-08-06T22-00-00`, in `Mario.load()` to check a specific timestamp.


## Project Structure
**train.py**
Training script that loops between Environment and Mario

**agent.py**
Defines a `Mario` class that helps the agent collect experiences, make actions given observations and updates the action policy.

**wrappers.py**
Defines environment pre-processing logic, including observation resizing, rgb to grayscale, etc.

**net.py**
Define Q-value estimators with a CNN.

**metrics.py**
Defines a `MetricLogger` class that helps track training/evaluation performance.

## Key Metrics

- Episode: current episode
- Step: total number of steps Mario played
- Epsilon: current exploration rate
- MeanReward: moving average of episode reward in past 100 episodes
- MeanLength: moving average of episode length in past 100 episodes
- MeanLoss: moving average of step loss in past 100 episodes
- MeanQValue: moving average of step Q value (predicted) in past 100 episodes

## Pre-trained

Checkpoint for a trained Mario: [INSERT LINK FOR SHARING TRAINED CHECKPOINT?]

## Resources

Deep Reinforcement Learning with Double Q-learning, Hado V. Hasselt et al, NIPS 2015: https://arxiv.org/abs/1509.06461

OpenAI Spinning Up tutorial: https://spinningup.openai.com/en/latest/

Reinforcement Learning: An Introduction, Richard S. Sutton et al. https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf

Deep Reinforcement Learning Doesn't Work Yet: https://www.alexirpan.com/2018/02/14/rl-hard.html
