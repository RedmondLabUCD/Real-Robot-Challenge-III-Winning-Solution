"""Example policy for Real Robot Challenge 2022"""
import torch
from rrc_2022_datasets import PolicyBase
import torch.nn as nn
import time

############################
model_name = 'bc_rnn_1000.pth'
############################


class BC(nn.Module):
    def __init__(self, 
                 obs_dim=97, 
                 action_dim = 9,
                 bias=True,
                 rnn_num_layers=2,
                 rnn = 'gru',):

        super(BC, self).__init__()
        self.max_action = 0.397
        
        if rnn == 'gru'"":
            self.rnn = nn.GRUCell(input_size = obs_dim,
                              hidden_size = 64, 
                              bias = bias)
            
        elif rnn == 'lstm':
            self.rnn = nn.LSTMCell(input_size = obs_dim, 
                              hidden_size = 64, 
                              bias = bias)
            
        self.predictor = nn.Sequential(nn.Linear(64, 256, bias=bias),
                                           nn.BatchNorm1d(256),
                                           nn.ReLU(),
                                           nn.Linear(256, 128, bias=bias),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128,action_dim, bias=bias),
                                           nn.Tanh(),
                                           )
        
    def forward(self,x,h):
        h = self.rnn(x,h)
        x = self.predictor(h) * self.max_action
        return x,h

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
            obs_dim = 139
        elif task_type == 'push':
            obs_dim = 97
        else:
            raise RuntimeError('The task type you input is invalid, only push and lift are avaliable')
        self.policy = BC(obs_dim=obs_dim, action_dim=9)
        self.policy.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.action_space = action_space
        self.h = torch.zeros(1,64).to(torch.device('cpu'))
        
    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        print('restting the env')
        self.h = torch.zeros(1,64).to(torch.device('cpu'))

    def get_action(self, observation):
        with torch.no_grad():
            self.policy.eval()
            observation = torch.Tensor([observation]).to(torch.float32)
            #t1 = time.time()
            action, self.h = self.policy(observation, self.h)
            action = action.cpu().detach().numpy()[0]
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
