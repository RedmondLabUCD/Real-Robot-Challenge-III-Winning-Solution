"""Example policy for Real Robot Challenge 2022"""
import torch
from rrc_2022_datasets import PolicyBase
import torch.nn as nn
import time
import numpy as np

############################
model_name = 'ckpt_cutted_bc3.pth'
############################


indexes_1 = range(111,111+9+1+9+9)
indexes_2 = range(59,59+24+4+3)

def lift_obs_cutter(obs):
    obs = np.delete(obs, indexes_1)
    obs = np.delete(obs, indexes_2)
    return obs

class BC(nn.Module):
    def __init__(self, 
                 obs_dim=97, 
                 action_dim = 9,
                 bias=True,
                 tune = False):

        super(BC_3, self).__init__()
        self.max_action = 0.397
        self.net = nn.Sequential(nn.Linear(obs_dim, 512, bias=bias),
                                 nn.BatchNorm1d(512),
                                           nn.ReLU(),
                                           nn.Linear(512, 512, bias=bias),
                                           nn.BatchNorm1d(512),
                                           nn.ReLU(),
                                           nn.Linear(512, 256, bias=bias),
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
        x = self.max_action * x
        return x

class TorchBasePolicy(PolicyBase):
    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
        model_path,
        task_type,
    ):
        self.action_space = action_space
        self.device = "cpu"
        if task_type == 'lift':
            obs_dim = 80
        elif task_type == 'push':
            obs_dim = 97
        else:
            raise RuntimeError('The task type you input is invalid, only push and lift are avaliable')
        self.policy = BC(obs_dim=obs_dim, action_dim=9)
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
            observation = torch.Tensor([lift_obs_cutter(observation)]).to(torch.float32)
            #t1 = time.time()
            action = self.policy(observation).cpu().detach().numpy()[0]
            #print(time.time() - t1)
            return action


class PushExpertPolicy(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model_path = f'/userhome/{model_name}'
        print('loading the expert pushing model from ', model_path)
        super().__init__(action_space, observation_space, episode_length, model_path, 'push')


class LiftExpertPolicy(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model_path = f'/userhome/{model_name}'
        print('loading the expert lifting model from ', model_path)
        super().__init__(action_space, observation_space, episode_length, model_path, 'lift')

class PushMixedPolicy(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model_path = f'/userhome/{model_name}'
        print('loading the mixed pushing model from ', model_path)
        super().__init__(action_space, observation_space, episode_length, model_path, 'push')


class LiftMixedPolicy(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        model_path = f'/userhome/{model_name}'
        print('loading the mixed lifting model from ', model_path)
        super().__init__(action_space, observation_space, episode_length, model_path, 'lift')
