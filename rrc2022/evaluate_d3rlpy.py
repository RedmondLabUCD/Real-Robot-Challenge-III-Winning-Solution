"""Example policy for Real Robot Challenge 2022"""
import numpy as np
import torch

#import tianshou
from rrc_2022_datasets import PolicyBase
from d3rlpy.dataset import MDPDataset
#from d3rlpy.algos import PLASWithPerturbation as algo
from d3rlpy.algos import PLAS as algo
import d3rlpy
from . import policies

model_name = 'model_925plas.pt'
json_name = 'params_925plas.json'

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

        # load torch script
        
        self.policy = algo.from_json(json_path)
        self.policy.load_model(model_path)
        self.action_space = action_space
        

    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        action = self.policy.predict([observation])[0]
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
