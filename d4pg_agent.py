import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import D4PGActor, D4PGCritic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-4       # learning rate of the critic
WEIGHT_DECAY = 0
NOISE_EPSILON = 1        # L2 weight decay
NOISE_DECAY = 0.999
REWARD_STEPS = 5
LEARN_EVERY_STEPS = 1
UPDATE_RATE = 1

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def categorical_distribution_projection(distribution_next_tensor, rewards_tensor, dones_tensor, gamma, device="cpu"):
    """Update critic network."""
    distribution_next = distribution_next_tensor.data.cpu().numpy()
    rewards = rewards_tensor.data.cpu().numpy()
    dones = dones_tensor.cpu().numpy().astype(np.bool)
    projected_distribution = np.zeros((len(rewards), N_ATOMS), dtype=np.float32)


    for atom in range(N_ATOMS):
        # calculate Bellman operator and clip the value in the range (Vmin, Vmax)
        tz = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma)) 
        # get the index of the projected value
        index = (tz - Vmin) / DELTA_Z  
        # Round down
        index_floor = np.floor(index).astype(np.int64)
        # Round up
        index_ceil = np.ceil(index).astype(np.int64)
        equal_mask = index_ceil == index_floor
        projected_distribution[equal_mask.squeeze(), index_floor[equal_mask]] += distribution_next[equal_mask.squeeze(), atom]
        # If the ceil lies between atoms, it is projected proporcionaly to the difference
        different_mask = index_ceil != index_floor
        
        projected_distribution[different_mask.squeeze(), index_floor[different_mask]] += distribution_next[different_mask.squeeze(), atom] * (index_ceil - index)[different_mask]
        projected_distribution[different_mask.squeeze(), index_ceil[different_mask]] += distribution_next[different_mask.squeeze(), atom] * (index - index_floor)[different_mask]

    
    if dones.any():
        projected_distribution[dones] = 0.0
        tz = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
        index = (tz - Vmin) / DELTA_Z
        index_floor = np.floor(index).astype(np.int64)
        index_ceil = np.ceil(index).astype(np.int64)
        equal_mask = index_ceil == index_floor
        equal_dones = dones.copy()
        equal_dones[dones] = equal_mask
        if equal_dones.any():
            projected_distribution[equal_dones, index_floor[equal_mask]] = 1.0
        different_mask = index_ceil != index_floor
        different_dones = dones.copy()
        different_dones[dones] = different_mask
        if different_dones.any():
            projected_distribution[different_dones, index_floor[different_mask]] = (index_ceil - index)[different_mask]
            projected_distribution[different_dones, index_ceil[different_mask]] = (index - index_floor)[different_mask]
    return torch.FloatTensor(projected_distribution).to(device)


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, num_agents = 1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): number of agents
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = D4PGActor(state_size, action_size, random_seed).to(device)
        self.actor_target = D4PGActor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        
        self.critic_local = D4PGCritic(state_size, action_size, random_seed).to(device)
        self.critic_target = D4PGCritic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)#, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_epsilon = NOISE_EPSILON

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, self.num_agents)
    
    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep % (LEARN_EVERY_STEPS) == 0:
            for _ in range(UPDATE_RATE):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise_epsilon * self.noise.sample()
            self.noise_epsilon *= NOISE_DECAY
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        critic_value_distribution = self.critic_local(states, actions) # Critic value distribution according to N_ATOMS
        actions_next = self.actor_target(next_states)
        distribution_values = self.critic_target(next_states, actions_next)
        distribution_next = F.softmax(distribution_values, dim=1) # Compute softmax funciton along rows 
        
        projected_distribution = categorical_distribution_projection(distribution_next, rewards, dones, gamma**REWARD_STEPS)

        probability_distribution = -F.log_softmax(critic_value_distribution, dim=1) * projected_distribution
        # Compute critic loss
        critic_loss = probability_distribution.sum(dim=1).mean()
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1) # Clip gradient
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions = self.actor_local(states)
        critic_value_distribution = self.critic_local(states, actions) # Critic value distribution according to N_ATOMS
        actor_loss = -self.critic_local.distr_to_q(critic_value_distribution)
        actor_loss = actor_loss.mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, num_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.num_agents = num_agents
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        for i in range(self.num_agents):
            e = self.experience(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])
            self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)