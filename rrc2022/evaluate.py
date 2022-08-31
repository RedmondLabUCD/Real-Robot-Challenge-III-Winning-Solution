"""Example policy for Real Robot Challenge 2022"""
import numpy as np
import torch

#import tianshou
from rrc_2022_datasets import PolicyBase
from d3rlpy.dataset import MDPDataset
#from d3rlpy.algos import PLASWithPerturbation as algo
from d3rlpy.algos import BC as algo
import d3rlpy
from . import policies

#obs = []
#act = []
#steps = 0
indexes_1 = range(111,111+9+1+9+9)
indexes_2 = range(59,59+1+1+24+4+3)

def obs_cutter(obs):
    obs = np.delete(obs, indexes_1)
    obs = np.delete(obs, indexes_2)
    return obs


############################
delete = 0
model_name = 'ckpt_300.pth'
############################


json_name = 'params_8261630.json'

import torch
import torch.nn as nn
import numpy as np
import gym

class BC(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, 
                 obs_dim=97, 
                 action_dim = 9,
                 bias=True):

        super(BC, self).__init__()
        self.max_action = 0.397
        self.net = nn.Sequential(nn.Linear(obs_dim, 256, bias=bias),
                                           nn.BatchNorm1d(256),
                                           nn.ReLU(),
                                           nn.Linear(256, 256, bias=bias),
                                           nn.BatchNorm1d(256),
                                           nn.ReLU(),
                                           nn.Linear(256, 128, bias=bias),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128,action_dim, bias=bias),
                                           nn.Tanh(),
                                           )
        
    def forward(self,x):
        x = self.net(x)
        # print(x)
        x = self.max_action * x
        # print(x)
        return x

class TorchBasePolicy(PolicyBase):
    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
        model_path,
        json_path,
    ):
        self.action_space = action_space
        self.device = "cpu"

        model_dim = np.load("/userhome/model_dim.npy",allow_pickle=True)
        obs_dim = model_dim[0]
        action_dim = model_dim[1]

        self.policy = BC(obs_dim=obs_dim, action_dim=action_dim)
        self.policy.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        
        self.action_space = action_space
        

    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        with torch.no_grad():
            self.policy.eval()
            observation = torch.Tensor([observation]).to(torch.float32)
            action = self.policy(observation).cpu().detach().numpy()[0]
            return action


class PushExpertPolicy(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model_path = f'/userhome/{model_name}'
        json_path = f'/userhome/{json_name}'
        print('loading the expert pushing model from ', model_path)
        super().__init__(action_space, observation_space, episode_length, model_path, json_path)


class LiftExpertPolicy(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model_path = f'/userhome/{model_name}'
        json_path = f'/userhome/{json_name}'
        print('loading the expert lifting model from ', model_path)
        super().__init__(action_space, observation_space, episode_length, model_path, json_path)

class PushMixedPolicy(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model_path = f'/userhome/{model_name}'
        json_path = f'/userhome/{json_name}'
        print('loading the mixed pushing model from ', model_path)
        super().__init__(action_space, observation_space, episode_length, model_path, json_path)


class LiftMixedPolicy(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model_path = f'/userhome/{model_name}'
        json_path = f'/userhome/{json_name}'
        print('loading the mixed lifting model from ', model_path)
        super().__init__(action_space, observation_space, episode_length, model_path, json_path)
