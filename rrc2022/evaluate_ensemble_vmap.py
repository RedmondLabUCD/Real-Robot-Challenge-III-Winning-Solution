"""Example policy for Real Robot Challenge 2022"""
import torch
from rrc_2022_datasets import PolicyBase
import torch.nn as nn
import numpy as np
from copy import copy
import time
from functorch import combine_state_for_ensemble
from functorch import vmap

class BC(nn.Module):
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
        x = self.max_action * x
        return x
    
############################
models_name = [
          'lm_s0.pth',
          'lm_s13.pth',
          'lm_s66.pth',
          'lm_s88.pth',
          'lm_s100.pth',
          'lm_s169.pth',
          'lm_s180.pth',
          'lm_s190.pth',
          'lm_s234.pth',
          'lm_s255.pth',
          'lm_rdm.pth',
         ]
############################

model_paths = []
for model in models_name:
    model_paths.append(f'/userhome/{model}')
    
models = []
for model_path in model_paths:
    model = BC(obs_dim=139, action_dim=9).to(torch.device('cpu'))
    model.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    models.append(copy(model))
    
fmodel, params, buffers = combine_state_for_ensemble(models)    

class TorchBasePolicy(PolicyBase):
    def __init__(
        self,
        action_space,
        observation_space,
        episode_length,
        model_paths,
        task_type,
    ):

        self.action_space = action_space
        
    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        # t1 = time.time()
        obs = torch.Tensor([observation]).to(torch.float32).to(torch.device('cpu'))
        actions = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, obs)
        # print(time.time() - t1)
        actions = np.array(copy(actions).cpu().detach().numpy())
        action = actions.mean(axis=0)[0]
        return action


class PushExpertPolicy(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        print('loading the expert pushing model from ', model_paths)
        super().__init__(action_space, observation_space, episode_length, model_paths, 'push')


class LiftExpertPolicy(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        print('loading the expert lifting model from ', model_paths)
        super().__init__(action_space, observation_space, episode_length, model_paths, 'lift')

class PushMixedPolicy(TorchBasePolicy):
    """Example policy for the push task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        print('loading the mixed pushing model from ', model_paths)
        super().__init__(action_space, observation_space, episode_length, model_paths, 'push')


class LiftMixedPolicy(TorchBasePolicy):
    """Example policy for the lift task, using a torch model to provide actions.
    Expects flattened observations.
    """

    def __init__(self, action_space, observation_space, episode_length):
        print('loading the mixed lifting model from ', model_paths)
        super().__init__(action_space, observation_space, episode_length, model_paths, 'lift')
