import torch
import random, numpy as np
from pathlib import Path
from net import MarioNet
import config
from collections import deque


# create a class Mario to represent the agent in the game
class Mario:
    def __init__(self, 
                state_dim, 
                action_dim, 
                save_dir, 
                memory, 
                exploration_rate_decay,
                learn_every,
                sync_every,
                checkpoint=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=memory)
        self.batch_size = config.BATCH_SIZE

        self.exploration_rate = config.EXPLORATION_RATE
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = config.EXPLORATION_RATE_MIN
        self.gamma = config.GAMMA

        self.curr_step = config.CURR_STEP
        self.burnin = config.BURNIN  # min. experiences before training
        self.learn_every = learn_every   # no. of experiences between updates to Q_online
        self.sync_every = sync_every   # no. of experiences between Q_target & Q_online sync

        self.save_every = config.SAVE_EVERY   # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.OPTIMIZER_LEARNING_RATE)
        self.loss_fn = config.LOSS_FN

    def act(self, state):
        """
        Based on the current state of the environment, 
        Mario acts according to the optimal action policy and updates step value

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        """
        # Explore (perform a random action)
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # Exploit (do the most optimal action-value policy)
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
 

    def store_to_memory(self, state, next_state, action, reward, done):
        """
        Each time Mario performs an action, he stores the experience to memory. 
        His experiece includes: 
        - current state, 
        - action performed, 
        - reward from the action, 
        - next state
        - whether game is done
        
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool)
        """
        if self.use_cuda:
            state = torch.FloatTensor(state).cuda()  
            next_state = torch.FloatTensor(next_state).cuda() 
            action = torch.LongTensor([action]).cuda()
            reward = torch.DoubleTensor([reward]).cuda()
            done = torch.BoolTensor([done]).cuda()
        else:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            action = torch.LongTensor([action])
            reward = torch.DoubleTensor([reward])
            done = torch.BoolTensor([done])

        self.memory.append((state, 
            next_state, 
            action, 
            reward, 
            done,)
        )


    def sample_from_memory(self):
        """
        Randomly sample a batch of experiences from memory and use it to learn the game
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def estimate_returns(self, state, action):
        """
        Compute estimate returns which is the predicted optimal Q* for a given state s
        where
        estimate_returns = Q_online*(s,a)

        Inputs:
        state (LazyFrame),
        action (int)
        """
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] 
        return current_Q

    # use the decorator '@torch.no_grad()' to disable gradient calculations here 
    # because we don’t need to backpropagate on params for Q_target
    @torch.no_grad()
    def target_returns(self, reward, next_state, done):
        """
        Compute target returns: aggregation of current reward and estimated Q* in next state s'
        where 
        a' = argmax Q_online(s',a)
        target_returns = current reward + gamma * Q_target*(s',a')

        Inputs:
        reward (float),
        next_state (LazyFrame),
        done(bool)
        """
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        # since we don't know what next action a' will be, 
        # we use the action the maximizes Q_online in next state s'
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action] 
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, estimate_returns, target_returns):
        """
        As Mario samples inputs from the replay buffer (memory) 
        the target_returns & estimate_returns are computed 
        and used to backpropagate loss down Q_online to update Q_online params
        
        UPDATED params for Q_online <- OLD params for Q_online + optimizer learning rate * d/dt(estimate_returns - target_returns)
        
        Inputs:
        estimate_returns (float),
        target_returns (float)
        """
        loss = self.loss_fn(estimate_returns, target_returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """
        Parameters for Q_target are not updated through backpropagation
        instead we periodically copy params from Q_online to Q_target
        """
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        """
        Learning step for Mario agent
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.sample_from_memory()

        # Compute estimate returns
        est = self.estimate_returns(state, action)

        # Compute target returns
        tgt = self.target_returns(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(est, tgt)

        return (est.mean().item(), loss)


    def save(self):
        """
        Saves MarioNet state dict
        """
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"[INFO] MarioNet saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        """
        Load the trained model checkpoint

        Inputs:
        load_path (str) eg. Path('checkpoints/2021-08-01T18-25-27/mario.chkpt')
        """
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"[INFO] Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
