import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, cfg, obs_shape, action_shape):
        self.device = cfg.device
        self.cfg = cfg
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((int(cfg.replay_buffer_capacity), *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((int(cfg.replay_buffer_capacity), *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((int(cfg.replay_buffer_capacity), *action_shape), dtype=np.float32)
        self.rewards = np.empty((int(cfg.replay_buffer_capacity), 1), dtype=np.float32)
        self.not_dones = np.empty((int(cfg.replay_buffer_capacity), 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((int(cfg.replay_buffer_capacity), 1), dtype=np.float32)

        self.idx = int(0)
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.cfg.replay_buffer_capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        print('idx:', self.idx)
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = int((self.idx + 1) % self.cfg.replay_buffer_capacity)
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.cfg.replay_buffer_capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max