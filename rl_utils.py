from tqdm import tqdm
import numpy as np
import torch
import collections
import random


from typing import Any, Callable, Dict, List, NamedTuple, Optional, SupportsFloat, Tuple, Union

# def generate_map_2d(map_flat, view=2):
#     size = 2 * view + 1
#     map_2d = np.array(map_flat).reshape((size, size))
#     return map_2d

# From stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_shape: int,
        device='cuda',
        gae_lambda: float = 1,
        gamma: float = 0.99,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size,) + self.action_shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,) + (1,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,)+ (1,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,)+ (1,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,)+ (1,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,)+ (1,), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray):
        last_values = last_values.detach().cpu().numpy().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        episode_start: bool
    ) -> None:
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = action.detach().cpu().numpy()
        self.rewards[self.pos] = reward.detach().cpu().numpy()
        self.values[self.pos] = value.detach().cpu().numpy()
        self.log_probs[self.pos] = log_prob.detach().cpu().numpy()
        self.episode_starts[self.pos] = episode_start
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def get(self, batch_size: Optional[int] = None):
        upper_bound = self.buffer_size if self.full else self.pos
        if batch_size is None:
            batch_size = upper_bound

        indices = np.random.permutation(upper_bound)
        for start in range(0, upper_bound, batch_size):
            end = min(start + batch_size, upper_bound)
            batch_indices = indices[start:end]
            yield self._get_samples(batch_indices)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> RolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

    def save(self, filename: str):
        np.savez(filename, 
                 observations=self.observations,
                 actions=self.actions,
                 rewards=self.rewards,
                 advantages=self.advantages,
                 returns=self.returns,
                 log_probs=self.log_probs,
                 values=self.values,
                 episode_starts=self.episode_starts,
                 pos=self.pos,
                 full=self.full)

    def load(self, filename: str):
        data = np.load(filename)
        self.observations = data['observations']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.advantages = data['advantages']
        self.returns = data['returns']
        self.log_probs = data['log_probs']
        self.values = data['values']
        self.episode_starts = data['episode_starts']
        self.pos = data['pos']
        self.full = data['full']


# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = collections.deque(maxlen=capacity) 

#     def add(self, state, action, reward, next_state, done): 
#         self.buffer.append((state, action, reward, next_state, done)) 

#     def sample(self, batch_size): 
#         transitions = random.sample(self.buffer, batch_size)
#         state, action, reward, next_state, done = zip(*transitions)
#         return np.array(state), action, reward, np.array(next_state), done 

#     def size(self): 
#         return len(self.buffer)

# def moving_average(a, window_size):
#     cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
#     middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
#     r = np.arange(1, window_size-1, 2)
#     begin = np.cumsum(a[:window_size-1])[::2] / r
#     end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
#     return np.concatenate((begin, middle, end))

# def train_on_policy_agent(env, agent, num_episodes):
#     return_list = []
#     for i in range(10):
#         with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
#             for i_episode in range(int(num_episodes/10)):
#                 episode_return = 0
#                 transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
#                 state = env.reset()[0]
#                 done = False
#                 truncated = False
#                 while not (done or truncated):
#                     action = agent.take_action(state)
#                     next_state, reward, done, truncated,_ = env.step(action)
#                     transition_dict['states'].append(state)
#                     transition_dict['actions'].append(action)
#                     transition_dict['next_states'].append(next_state)
#                     transition_dict['rewards'].append(reward)
#                     transition_dict['dones'].append(done)
#                     state = next_state
#                     episode_return += reward
#                 return_list.append(episode_return)
#                 agent.update(transition_dict)
#                 if (i_episode+1) % 10 == 0:
#                     pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
#                 pbar.update(1)
#     return return_list

# def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
#     return_list = []
#     for i in range(10):
#         with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
#             for i_episode in range(int(num_episodes/10)):
#                 episode_return = 0
#                 state = env.reset()
#                 done = False
#                 while not done:
#                     action = agent.take_action(state)
#                     next_state, reward, done, _ = env.step(action)
#                     replay_buffer.add(state, action, reward, next_state, done)
#                     state = next_state
#                     episode_return += reward
#                     if replay_buffer.size() > minimal_size:
#                         b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
#                         transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
#                         agent.update(transition_dict)
#                 return_list.append(episode_return)
#                 if (i_episode+1) % 10 == 0:
#                     pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
#                 pbar.update(1)
#     return return_list


# def compute_advantage(gamma, lmbda, td_delta):
#     td_delta = td_delta.detach().numpy()
#     advantage_list = []
#     advantage = 0.0
#     for delta in td_delta[::-1]:
#         advantage = gamma * lmbda * advantage + delta
#         advantage_list.append(advantage)
#     advantage_list.reverse()
#     return np.array(advantage_list)