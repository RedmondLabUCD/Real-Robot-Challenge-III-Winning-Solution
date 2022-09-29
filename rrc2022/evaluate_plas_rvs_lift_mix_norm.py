import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from rrc_2022_datasets import PolicyBase
import time

filename='model_9290856'
normname = 'norm_9290856.npy'
directory = '/userhome'

device = torch.device("cpu")

class Normlizor():        
    def norm(self, obs):
        return ((obs - self.mean) / (self.std))
    
    def norm_goal(self, goal):
        return ((goal - self.goal_mean) / (self.goal_std))

    def load(self, norm_path):
        norm = np.load(norm_path, allow_pickle=True).item()
        self.min = norm['min']
        self.max = norm['max']
        self.mean = norm['mean']
        self.std = norm['std']
        self.goal_min = norm['goal_min']
        self.goal_max = norm['goal_max']
        self.goal_mean = norm['goal_mean']
        self.goal_std = norm['goal_std']

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.hidden_size = (400, 300)

        self.l1 = nn.Linear(state_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.hidden_size = (400, 300)

        self.l1 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], 1)

        self.l4 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l5 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l6 = nn.Linear(self.hidden_size[1], 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_size=750):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None, clip=None, raw=False):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(device)
            if clip is not None:
                z = z.clamp(-clip, clip)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
        if raw: return a
        return self.max_action * torch.tanh(a)


class VAEModule(object):
    def __init__(self, *args, vae_lr=1e-4, **kwargs):
        self.vae = VAE(*args, **kwargs).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)

    def load(self, filename, directory):
        print('%s/%s_vae.pth' % (directory, filename))
        self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=torch.device('cpu')))


class Latent(object):
    def __init__(self, vae, state_dim, action_dim, latent_dim, max_action, discount=0.99, tau=0.005,
                 actor_lr=1e-3, critic_lr=1e-3, lmbda=0.75, max_latent_action=2, **kwargs):
        self.actor = Actor(state_dim, latent_dim, max_latent_action).to(device)

        self.latent_dim = latent_dim
        self.vae = vae
        self.max_action = max_action
        self.max_latent_action = max_latent_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.vae.decode(state, z=self.actor(state))
        return action.cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename),map_location=torch.device('cpu')))

class TorchBasePolicy(PolicyBase):
    def __init__(
        self,
        action_space,
        observation_space,
        episode_length
    ):
        self.action_space = action_space
        self.device = "cpu"

        # load torch script
        
        vae_trainer = VAEModule(state_dim=139+24, action_dim=9, latent_dim=18, max_action=0.397, vae_lr=1e-4, hidden_size=750)
        vae_trainer.load(filename=filename, directory=directory)
        
        self.normlizor = Normlizor()
        self.normlizor.load(f"{directory}/{normname}")
        self.policy = Latent(vae_trainer.vae, state_dim=139+24, action_dim=9, latent_dim=18, max_action=0.397)
        self.policy.load(filename=filename, directory=directory)
        self.action_space = action_space
        
    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        normed_obs = np.array(self.normlizor.norm(observation))
        normed_goal = np.array(self.normlizor.norm_goal(observation[33:33+24]))
        observation = np.concatenate((normed_obs, normed_goal))
        action = self.policy.select_action(observation)
        print(action)
        return action

class LiftMixedPolicy(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.
    Expects flattened observations.
    """
    def __init__(self, action_space, observation_space, episode_length):
        super().__init__(action_space, observation_space, episode_length)
