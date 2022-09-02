"""Example policy for Real Robot Challenge 2022"""
import torch
from rrc_2022_datasets import PolicyBase
import torch.nn as nn
import numpy as np
from copy import copy
import time

############################
models = [
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
for model in models:
    model_paths.append(f'/userhome/{model}')

class ensumble():
    def __init__(self,
                 models,
                 weight_ratio=0.1,
                 device=torch.device('cpu')):
        self.models = models
        self.weight = [1] * len(self.models)
        self.weight_ratio = weight_ratio
        self.device = device
        
    def cta_action(self, obs):
        actions = self.gen_actions(obs)
        temp_actions = np.array(copy(actions))
        avg = temp_actions.mean(axis=0)
        for idx,action in enumerate(actions):
            if idx == 0:
                dis = np.linalg.norm(action - avg)
                action_to_apply = action
                action_to_apply_idx = 0
            else:
                if np.linalg.norm(action - avg) <= dis:
                    dis = np.linalg.norm(action - avg)
                    action_to_apply = action
                    action_to_apply_idx = idx
        return action_to_apply, action_to_apply_idx
    
    def avg_action(self, obs):
        actions = self.gen_actions(obs)
        temp_actions = np.array(copy(actions))
        return temp_actions.mean(axis=0), -1
    
    def weight_avg_action(self, obs):
        actions = self.gen_actions(obs)
        temp_actions = np.array(copy(actions))
        avg = temp_actions.mean(axis=0)
        for idx,action in enumerate(actions):
            if idx == 0:
                dis = np.linalg.norm(action - avg)
                action_to_apply_idx = 0
            else:
                if np.linalg.norm(action - avg) <= dis:
                    dis = np.linalg.norm(action - avg)
                    action_to_apply_idx = idx
        self.weight[idx] += self.weight_ratio
        return np.average(temp_actions, axis=0,weights=self.weight), action_to_apply_idx
    
    def gen_actions(self, obs):
        obs = torch.Tensor([obs]).to(torch.float32).to(self.device)
        actions = []
        for model in self.models:
            model.eval()
            actions.append(model(obs).cpu().detach().numpy()[0])
        return actions
    
    def reset(self):
        self.weight = [1] * len(self.models)

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
        self.device = "cpu"
        if task_type == 'lift':
            obs_dim = 139
        elif task_type == 'push':
            obs_dim = 97
        else:
            raise RuntimeError('The task type you input is invalid, only push and lift are avaliable')
        
        self.policys = []
        for model_path in model_paths:
            policy = BC(obs_dim=obs_dim, action_dim=9)
            policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.policys.append(copy(policy))
        self.action_space = action_space
        self.esb = ensumble(self.policys, weight_ratio=2)
        
    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        with torch.no_grad():
            #observation = torch.Tensor([observation]).to(torch.float32)
            action,_ = self.esb.avg_action(obs)
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
