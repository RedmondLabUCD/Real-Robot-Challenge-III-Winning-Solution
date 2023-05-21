#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Example policy for Real Robot Challenge 2022"""
import torch
from rrc_2022_datasets import PolicyBase
#from d3rlpy.algos import BC as algo
#from d3rlpy.algos import PLAS as algo
#from d3rlpy.algos import CRR as algo
#from d3rlpy.algos import TD3PlusBC as algo
from d3rlpy.algos import IQL as algo

model_name = 'iql_mixed_ok.pt'
json_name = 'iql_mixed_ok.json'

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

        self.policy = algo.from_json(json_path)
        self.policy.load_model(model_path)
        self.action_space = action_space
        
    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        pass
    
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
