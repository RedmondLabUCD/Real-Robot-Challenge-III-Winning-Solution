#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 21:46:33 2022

@author: qiang
"""

import gym
import numpy as np
from copy import copy
from torch.utils.data import Dataset, DataLoader
import rrc_2022_datasets
import os
from utils import rot_augment, Normlizor
import math
from tqdm.auto import tqdm
import random
import torch
import gc
import time
from d3rlpy.dataset import MDPDataset

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def train_tune_dataset_process(aug_path, tune_path):
    norm = Normlizor()
    aug_dataset = np.load(aug_path, allow_pickle=True).item()
    norm.init_norm_with_dataset(aug_dataset, 'std', name='train_aug_norm_params', save_path=os.path.split(aug_path)[0])
    dataset = MDPDataset(aug_dataset["observations"],
                         aug_dataset["actions"],
                         aug_dataset["rewards"],
                         aug_dataset["timeouts"].astype(np.int32))
    dataset.dump(os.path.join(os.path.split(aug_path)[0], 'train_aug_dataset.h5'))
    del aug_dataset, dataset
    gc.collect
    
    tune_dataset = np.load(tune_path, allow_pickle=True).item()
    dataset = MDPDataset(tune_dataset["observations"],
                         tune_dataset["actions"],
                         tune_dataset["rewards"],
                         tune_dataset["timeouts"].astype(np.int32))
    dataset.dump(os.path.join(os.path.split(tune_path)[0], 'tune_dataset.h5'))
    del tune_dataset, dataset
    gc.collect

def load_rrc_raw_dataset(args):
    if args.task == 'real_push_exp':
        task_name = "trifinger-cube-push-real-expert-v0"
        print("Training the real pushing task with expert dataset")
    elif args.task == 'real_lift_exp':
        task_name = "trifinger-cube-lift-real-expert-v0"
        print("Training the real lifting task with expert dataset")
    elif args.task == 'real_push_mix':
        task_name = "trifinger-cube-push-real-mixed-v0"
        print("Training the real pushing task with mixed dataset")
    elif args.task == 'real_lift_mix':
        task_name = "trifinger-cube-lift-real-mixed-v0"
        print("Training the real lifting task with mixed dataset")
    else:
        raise RuntimeError(
            'The task name you input is invalid, only push and lift are avaliable')

    if args.raw_dataset_path:
        raw_dataset = np.load(args.raw_dataset_path, allow_pickle=True).item()
    else:
        env = gym.make(
            task_name,
            disable_env_checker=True,
        )
        gym_dataset = env.get_dataset()
        raw_dataset = {}
        raw_dataset['observations'] = gym_dataset['observations']
        raw_dataset['rewards'] = gym_dataset['rewards']
        raw_dataset['timeouts'] = gym_dataset['timeouts']
        raw_dataset['actions'] = gym_dataset['actions']
        del gym_dataset
        gc.collect()

    return raw_dataset


def filter_by_net(dataset, contra_filter, conf, args):
    overall_length = dataset['timeouts'].shape[0]
    good_indexes = []
    good_index_count = 0
    for filtered_data_type in ['positive', 'negative']:
        temp_obs = []
        temp_actions = []
        temp_rewards = []
        temp_timeouts = []
        first_flag = True
        temp_first_flag = True
        temp_obs_numpy = None
        temp_actions_numpy = None
        temp_rewards_numpy = None
        temp_timeouts_numpy = None
        seq_num = 0
        for idx, timeout in enumerate(dataset['timeouts']):
            temp_obs.append(dataset['observations'][idx].tolist())
            temp_actions.append(dataset['actions'][idx].tolist())
            temp_rewards.append(dataset['rewards'][idx].tolist())
            temp_timeouts.append(dataset['timeouts'][idx].tolist())

            if timeout:
                if torch.cuda.is_available() and args.use_gpu:
                    temp_obs_tensor = torch.tensor(temp_obs.copy()).cuda(
                        non_blocking=True).to(torch.float32)
                    temp_actions_tensor = torch.tensor(temp_actions.copy()).cuda(
                        non_blocking=True).to(torch.float32)
                else:
                    temp_obs_tensor = torch.tensor(temp_obs.copy()).to(
                        torch.float32).to(torch.device('cpu'))
                    temp_actions_tensor = torch.tensor(temp_actions.copy()).to(
                        torch.float32).to(torch.device('cpu'))

                prob = contra_filter(temp_obs_tensor, temp_actions_tensor).cpu(
                ).detach().numpy()[..., 0].sum()

                if filtered_data_type == 'positive':
                    cond = (prob >= conf)
                else:
                    cond = not (prob >= conf)

                if cond:
                    if first_flag:
                        obs = np.array(temp_obs)
                        actions = np.array(temp_actions)
                        rewards = np.array(temp_rewards)
                        timeouts = np.array(temp_timeouts)
                        seq_num += 1
                        first_flag = False
                    else:
                        if seq_num % 100 == 0:
                            obs = np.concatenate((obs, temp_obs_numpy))
                            actions = np.concatenate(
                                (actions, temp_actions_numpy))
                            rewards = np.concatenate(
                                (rewards, temp_rewards_numpy))
                            timeouts = np.concatenate(
                                (timeouts, temp_timeouts_numpy))
                            temp_obs_numpy = np.array(temp_obs)
                            temp_actions_numpy = np.array(temp_actions)
                            temp_rewards_numpy = np.array(temp_rewards)
                            temp_timeouts_numpy = np.array(temp_timeouts)
                            seq_num += 1
                        else:
                            if temp_first_flag:
                                temp_obs_numpy = np.array(temp_obs)
                                temp_actions_numpy = np.array(temp_actions)
                                temp_rewards_numpy = np.array(temp_rewards)
                                temp_timeouts_numpy = np.array(temp_timeouts)
                                temp_first_flag = False
                            temp_obs_numpy = np.concatenate(
                                (temp_obs_numpy, np.array(temp_obs)))
                            temp_actions_numpy = np.concatenate(
                                (temp_actions_numpy, np.array(temp_actions)))
                            temp_rewards_numpy = np.concatenate(
                                (temp_rewards_numpy, np.array(temp_rewards)))
                            temp_timeouts_numpy = np.concatenate(
                                (temp_timeouts_numpy, np.array(temp_timeouts)))
                            seq_num += 1

                    if filtered_data_type == 'positive':
                        good_indexes.append(good_index_count)

                if filtered_data_type == 'positive':
                    good_index_count += 1

                temp_obs = []
                temp_actions = []
                temp_rewards = []
                temp_timeouts = []

            if idx % 5e5 == 0 and filtered_data_type == 'positive':
                print(
                    f'Filtering {args.task} dataset by contra-filter, Turn {args.turn_num}, POSITIVE progress: {idx} / {overall_length}')
            elif idx % 5e5 == 0 and filtered_data_type == 'negative':
                print(
                    f'Filtering {args.task} dataset by contra-filter,Turn {args.turn_num} ,NEGATIVE progress: {idx} / {overall_length}')

        obs = np.concatenate((obs, temp_obs_numpy))
        actions = np.concatenate((actions, temp_actions_numpy))
        rewards = np.concatenate((rewards, temp_rewards_numpy))
        timeouts = np.concatenate((timeouts, temp_timeouts_numpy))
        temp_dataset = {}
        temp_dataset['observations'] = obs
        temp_dataset['actions'] = actions
        temp_dataset['rewards'] = rewards
        temp_dataset['timeouts'] = timeouts
        print(
            f'Saving to {args.save_path}/datasets/turn{args.turn_num}_{filtered_data_type}.npy')
        np.save(
            f'{args.save_path}/datasets/turn{args.turn_num}_{filtered_data_type}.npy', temp_dataset)
        del obs, actions, rewards, timeouts, temp_dataset
        gc.collect
    np.save(
        f'{args.save_path}/datasets/turn{args.turn_num}_good_indexes.npy', good_indexes)
    return good_indexes


def filter_by_reward(dataset, args):
    if args.task_type == 'push':
        reward_th = 0.98
        reward_th_len = 150
        start_reward_th = 0.33
        start_reward_th_len = 5
    elif args.task_type == 'lift':
        reward_th = 0.96
        reward_th_len = 150
        start_reward_th = 0.33
        start_reward_th_len = 5
    else:
        raise RuntimeError(
            'The input task type is invalid')
    overall_length = dataset['timeouts'].shape[0]

    good_indexes = []
    good_index_count = 0
    for filtered_data_type in ['positive', 'negative']:
        first_flag = True
        temp_first_flag = True
        temp_obs_numpy = None
        temp_actions_numpy = None
        temp_rewards_numpy = None
        temp_timeouts_numpy = None
        temp_obs = []
        temp_actions = []
        temp_rewards = []
        temp_timeouts = []
        seq_num = 0
        obs = []
        rewards = []
        timeouts = []
        actions = []
        if filtered_data_type == 'positive':  # Computation in list is quicker
            for idx, timeout in enumerate(dataset['timeouts']):
                temp_obs.append(dataset['observations'][idx].tolist())
                temp_rewards.append(dataset['rewards'][idx])
                temp_timeouts.append(dataset['timeouts'][idx])
                temp_actions.append(dataset['actions'][idx].tolist())
                if timeout:
                    cond = (np.array(copy(temp_rewards[-reward_th_len:])).mean() >= reward_th
                            and np.array(copy(temp_rewards[0:start_reward_th_len])).mean() <= start_reward_th)
                    if cond:
                        obs += temp_obs
                        rewards += temp_rewards
                        timeouts += temp_timeouts
                        actions += temp_actions
                        good_indexes.append(good_index_count)
                    temp_obs = []
                    temp_rewards = []
                    temp_timeouts = []
                    temp_actions = []
                    good_index_count += 1
                if idx % 5e5 == 0:
                    print(
                        f'Filtering {args.task} dataset by reward, POSITIVE part, progress: {idx} / {overall_length}')
            temp_dataset = {}
            temp_dataset['observations'] = np.array(obs)
            temp_dataset['actions'] = np.array(actions)
            temp_dataset['rewards'] = np.array(rewards)
            temp_dataset['timeouts'] = np.array(timeouts)

        elif filtered_data_type == 'negative':  # Computation in numpy saves memory but slow
            for idx, timeout in enumerate(dataset['timeouts']):
                temp_obs.append(dataset['observations'][idx].tolist())
                temp_actions.append(dataset['actions'][idx].tolist())
                temp_rewards.append(dataset['rewards'][idx].tolist())
                temp_timeouts.append(dataset['timeouts'][idx].tolist())
                if timeout:
                    cond = not (np.array(copy(temp_rewards[-reward_th_len:])).mean() >= reward_th
                                and np.array(copy(temp_rewards[0:start_reward_th_len])).mean() <= start_reward_th)
                    if cond:
                        if first_flag:
                            obs = np.array(temp_obs)
                            actions = np.array(temp_actions)
                            rewards = np.array(temp_rewards)
                            timeouts = np.array(temp_timeouts)
                            seq_num += 1
                            first_flag = False
                        else:
                            if seq_num % 100 == 0:
                                obs = np.concatenate(
                                    (obs, temp_obs_numpy))
                                actions = np.concatenate(
                                    (actions, temp_actions_numpy))
                                rewards = np.concatenate(
                                    (rewards, temp_rewards_numpy))
                                timeouts = np.concatenate(
                                    (timeouts, temp_timeouts_numpy))
                                temp_obs_numpy = np.array(temp_obs)
                                temp_actions_numpy = np.array(temp_actions)
                                temp_rewards_numpy = np.array(temp_rewards)
                                temp_timeouts_numpy = np.array(temp_timeouts)
                                seq_num += 1
                            else:
                                if temp_first_flag:
                                    temp_obs_numpy = np.array(temp_obs)
                                    temp_actions_numpy = np.array(temp_actions)
                                    temp_rewards_numpy = np.array(temp_rewards)
                                    temp_timeouts_numpy = np.array(
                                        temp_timeouts)
                                    temp_first_flag = False
                                temp_obs_numpy = np.concatenate(
                                    (temp_obs_numpy, np.array(temp_obs)))
                                temp_actions_numpy = np.concatenate(
                                    (temp_actions_numpy, np.array(temp_actions)))
                                temp_rewards_numpy = np.concatenate(
                                    (temp_rewards_numpy, np.array(temp_rewards)))
                                temp_timeouts_numpy = np.concatenate(
                                    (temp_timeouts_numpy, np.array(temp_timeouts)))
                                seq_num += 1
                        if filtered_data_type == 'positive':
                            good_indexes.append(good_index_count)
                    if filtered_data_type == 'positive':
                        good_index_count += 1

                    temp_obs = []
                    temp_actions = []
                    temp_rewards = []
                    temp_timeouts = []
                if idx % 5e5 == 0 and filtered_data_type == 'negative':
                    print(
                        f'Filtering {args.task} dataset by reward, NEGATIVE part, progress: {idx} / {overall_length}')
            obs = np.concatenate((obs, temp_obs_numpy))
            actions = np.concatenate((actions, temp_actions_numpy))
            rewards = np.concatenate((rewards, temp_rewards_numpy))
            timeouts = np.concatenate((timeouts, temp_timeouts_numpy))
            temp_dataset = {}
            temp_dataset['observations'] = obs
            temp_dataset['actions'] = actions
            temp_dataset['rewards'] = rewards
            temp_dataset['timeouts'] = timeouts

        print(
            f'Saving to {args.save_path}/datasets/turn{args.turn_num}_{filtered_data_type}.npy')
        np.save(
            f'{args.save_path}/datasets/turn{args.turn_num}_{filtered_data_type}.npy', temp_dataset)
        del obs, actions, rewards, timeouts, temp_dataset
        gc.collect
    np.save(
        f'{args.save_path}/datasets/turn{args.turn_num}_good_indexes.npy', good_indexes)
    return good_indexes


def torch_loader(filtered_dataset,
                 raw_dataset,
                 train_batch_size,
                 shuffle,
                 ):
    filtered_dataset = np.load(filtered_dataset, allow_pickle=True).item()
    raw_dataset = np.load(raw_dataset, allow_pickle=True).item()
    dataset = TorchDatasetHandler(
        filtered_dataset, raw_dataset, transform=None)

    train_loader = DataLoader(
        dataset, batch_size=train_batch_size, shuffle=shuffle)
    return train_loader


class TorchDatasetHandler(Dataset):
    def __init__(self, filtered_dataset, raw_dataset, transform=None):
        self.dataset = filtered_dataset
        self.transform = transform
        self.raw_dataset = raw_dataset
        self.raw_len = self.raw_dataset['observations'].shape[0]
        print('Using loader Final')

    def __len__(self):
        return self.dataset['actions'].shape[0]

    def __getitem__(self, index):
        pos_observation = self.dataset['observations'][index]
        pos_action = self.dataset['actions'][index]

        if np.random.uniform(0, 1) <= 0.5:
            action = pos_action
            observation = pos_observation
            label = np.array([1, 0])
        else:
            label = np.array([0, 1])
            observation = pos_observation
            if np.random.uniform(0, 1) < 0.5:  # Random actions
                while True:
                    random_action = np.random.uniform(-0.397, 0.397, size=9)
                    if np.linalg.norm(random_action-pos_action) >= 0.4:
                        action = random_action
                        break
            else:
                while True:
                    random_num = np.random.randint(0, self.raw_len)
                    if np.linalg.norm(pos_action-self.raw_dataset["actions"][random_num]) >= 0.4:
                        action = self.raw_dataset["actions"][random_num]
                        break

        if self.transform:
            observation = self.transform(observation)
            action = self.transform(action)
            label = self.transform(label)

        data = [observation, action, label]
        return data


# This only works for rrc dataset, for saving the memory
def concatenate_datasets(datasets_list, save_path=None):
    data_0 = np.load(datasets_list[0], allow_pickle=True).item()
    data_1 = np.load(datasets_list[1], allow_pickle=True).item()
    data_2 = np.load(datasets_list[2], allow_pickle=True).item()

    obs = np.concatenate((data_0['observations'], data_1['observations']))
    rewards = np.concatenate((data_0['rewards'], data_1['rewards']))
    actions = np.concatenate((data_0['actions'], data_1['actions']))
    timeouts = np.concatenate((data_0['timeouts'], data_1['timeouts']))

    del data_0, data_1
    gc.collect()

    obs = np.concatenate((obs, data_2['observations']))
    rewards = np.concatenate((rewards, data_2['rewards']))
    actions = np.concatenate((actions, data_2['actions']))
    timeouts = np.concatenate((timeouts, data_2['timeouts']))

    del data_2
    gc.collect()

    temp_data = {}
    temp_data['observations'] = obs
    temp_data['rewards'] = rewards
    temp_data['actions'] = actions
    temp_data['timeouts'] = timeouts

    if not save_path:
        save_path = datasets_list[0].split('.')[0] + '_aug.npy'

    np.save(save_path, temp_data)

    time.sleep(2)

    del temp_data
    gc.collect()


def symmetry_lift_aug_cw120(raw_dataset, save_path):
    env = gym.make(
        "trifinger-cube-lift-sim-expert-v0",
    )
    env.reset()
    angle = math.radians(-120)
    obs = []

    with tqdm(total=raw_dataset['observations'].shape[0], desc="Augmenting the lift task cw120") as pbar:
        for idx, obs in enumerate(raw_dataset['observations']):

            temp_obs = copy(obs)

            # observations
            # ag
            temp_obs[0:0+2] = rot_augment(obs[0:0+2], angle)
            temp_obs[3:3+2] = rot_augment(obs[3:3+2], angle)
            temp_obs[6:6+2] = rot_augment(obs[6:6+2], angle)
            temp_obs[9:9+2] = rot_augment(obs[9:9+2], angle)
            temp_obs[12:12+2] = rot_augment(obs[12:12+2], angle)
            temp_obs[15:15+2] = rot_augment(obs[15:15+2], angle)
            temp_obs[18:18+2] = rot_augment(obs[18:18+2], angle)
            temp_obs[21:21+2] = rot_augment(obs[21:21+2], angle)

            # act
            _act_angle_0 = copy(obs[24:24+3])
            _act_angle_120 = copy(obs[27:27+3])
            _act_angle_240 = copy(obs[30:30+3])
            temp_obs[24:24+3] = _act_angle_240
            temp_obs[27:27+3] = _act_angle_0
            temp_obs[30:30+3] = _act_angle_120

            # g
            temp_obs[33:33+2] = rot_augment(obs[33:33+2], angle)
            temp_obs[36:36+2] = rot_augment(obs[36:36+2], angle)
            temp_obs[39:39+2] = rot_augment(obs[39:39+2], angle)
            temp_obs[42:42+2] = rot_augment(obs[42:42+2], angle)
            temp_obs[45:45+2] = rot_augment(obs[45:45+2], angle)
            temp_obs[48:48+2] = rot_augment(obs[48:48+2], angle)
            temp_obs[51:51+2] = rot_augment(obs[51:51+2], angle)
            temp_obs[54:54+2] = rot_augment(obs[54:54+2], angle)

            # confidence not change 57
            # delay not change 58

            # ag2
            temp_obs[59:59+24] = temp_obs[0:0+24]

            #orientation and position
            ag_key_points = copy(temp_obs[0:0+24]).reshape(8, 3)
            ag_pos = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[
                0]
            ag_ori = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[
                1]
            temp_obs[83:83+4] = ag_ori
            temp_obs[87:87+3] = ag_pos

            # rob pos
            _rpos_angle_0 = copy(obs[111:111+3])
            _rpos_angle_120 = copy(obs[114:114+3])
            _rpos_angle_240 = copy(obs[117:117+3])
            temp_obs[111:111+3] = _rpos_angle_240
            temp_obs[114:114+3] = _rpos_angle_0
            temp_obs[117:117+3] = _rpos_angle_120

            # rob id not change 120

            # rob torque
            _rtorque_angle_0 = copy(obs[121:121+3])
            _rtorque_angle_120 = copy(obs[124:124+3])
            _rtorque_angle_240 = copy(obs[127:127+3])
            temp_obs[121:121+3] = _rtorque_angle_240
            temp_obs[124:124+3] = _rtorque_angle_0
            temp_obs[127:127+3] = _rtorque_angle_120

            # rob vel
            _rvel_angle_0 = copy(obs[130:130+3])
            _rvel_angle_120 = copy(obs[133:133+3])
            _rvel_angle_240 = copy(obs[136:136+3])
            temp_obs[130:130+3] = _rvel_angle_240
            temp_obs[133:133+3] = _rvel_angle_0
            temp_obs[136:136+3] = _rvel_angle_120

            # finger tip force
            _f_force_angle_0 = copy(obs[90])
            _f_force_angle_120 = copy(obs[91])
            _f_force_angle_240 = copy(obs[92])
            temp_obs[90] = _f_force_angle_240
            temp_obs[91] = _f_force_angle_0
            temp_obs[92] = _f_force_angle_120

            # finger tip pos and vel
            fingertip_position, fingertip_velocity = env.sim_env.platform.forward_kinematics(
                temp_obs[111:111+9], temp_obs[130:130+9]
            )
            fingertip_position = np.array(fingertip_position, dtype=np.float32)
            fingertip_velocity = np.array(fingertip_velocity, dtype=np.float32)
            temp_obs[93:93+9] = fingertip_position.reshape(9,)
            temp_obs[102:102+9] = fingertip_velocity.reshape(9,)

            raw_dataset['observations'][idx] = temp_obs

            # action
            _action_0 = copy(raw_dataset['actions'][idx][0:0+3])
            _action_120 = copy(raw_dataset['actions'][idx][3:3+3])
            _action_240 = copy(raw_dataset['actions'][idx][6:6+3])
            raw_dataset['actions'][idx][0:0+3] = _action_240
            raw_dataset['actions'][idx][3:3+3] = _action_0
            raw_dataset['actions'][idx][6:6+3] = _action_120

            if idx % 100000 == 0:
                pbar.set_postfix()
                pbar.update(100000)

    np.save(
        save_path, raw_dataset)
    print('Scuuessfuly saved')
    
    del raw_dataset
    gc.collect()


def symmetry_lift_aug_ccw120(raw_dataset, save_path):
    env = gym.make(
        "trifinger-cube-lift-sim-expert-v0",
    )
    env.reset()
    angle = math.radians(-240)
    obs = []

    with tqdm(total=raw_dataset['observations'].shape[0], desc="Augmenting the lift task ccw120") as pbar:
        for idx, obs in enumerate(raw_dataset['observations']):

            if idx % 100000 == 0:
                pbar.update(10)

            temp_obs = copy(obs)
            # observations
            # ag
            temp_obs[0:0+2] = rot_augment(obs[0:0+2], angle)
            temp_obs[3:3+2] = rot_augment(obs[3:3+2], angle)
            temp_obs[6:6+2] = rot_augment(obs[6:6+2], angle)
            temp_obs[9:9+2] = rot_augment(obs[9:9+2], angle)
            temp_obs[12:12+2] = rot_augment(obs[12:12+2], angle)
            temp_obs[15:15+2] = rot_augment(obs[15:15+2], angle)
            temp_obs[18:18+2] = rot_augment(obs[18:18+2], angle)
            temp_obs[21:21+2] = rot_augment(obs[21:21+2], angle)

            # act
            _act_angle_0 = copy(obs[24:24+3])
            _act_angle_120 = copy(obs[27:27+3])
            _act_angle_240 = copy(obs[30:30+3])
            temp_obs[24:24+3] = _act_angle_120
            temp_obs[27:27+3] = _act_angle_240
            temp_obs[30:30+3] = _act_angle_0

            # g
            temp_obs[33:33+2] = rot_augment(obs[33:33+2], angle)
            temp_obs[36:36+2] = rot_augment(obs[36:36+2], angle)
            temp_obs[39:39+2] = rot_augment(obs[39:39+2], angle)
            temp_obs[42:42+2] = rot_augment(obs[42:42+2], angle)
            temp_obs[45:45+2] = rot_augment(obs[45:45+2], angle)
            temp_obs[48:48+2] = rot_augment(obs[48:48+2], angle)
            temp_obs[51:51+2] = rot_augment(obs[51:51+2], angle)
            temp_obs[54:54+2] = rot_augment(obs[54:54+2], angle)

            # confidence not change 57
            # delay not change 58

            # ag2
            temp_obs[59:59+24] = temp_obs[0:0+24]

            #orientation and position
            ag_key_points = copy(temp_obs[0:0+24]).reshape(8, 3)
            ag_pos = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[
                0]
            ag_ori = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[
                1]
            temp_obs[83:83+4] = ag_ori
            temp_obs[87:87+3] = ag_pos

            # rob pos
            _rpos_angle_0 = copy(obs[111:111+3])
            _rpos_angle_120 = copy(obs[114:114+3])
            _rpos_angle_240 = copy(obs[117:117+3])
            temp_obs[111:111+3] = _rpos_angle_120
            temp_obs[114:114+3] = _rpos_angle_240
            temp_obs[117:117+3] = _rpos_angle_0

            # rob id not change 120

            # rob torque
            _rtorque_angle_0 = copy(obs[121:121+3])
            _rtorque_angle_120 = copy(obs[124:124+3])
            _rtorque_angle_240 = copy(obs[127:127+3])
            temp_obs[121:121+3] = _rtorque_angle_120
            temp_obs[124:124+3] = _rtorque_angle_240
            temp_obs[127:127+3] = _rtorque_angle_0

            # rob vel
            _rvel_angle_0 = copy(obs[130:130+3])
            _rvel_angle_120 = copy(obs[133:133+3])
            _rvel_angle_240 = copy(obs[136:136+3])
            temp_obs[130:130+3] = _rvel_angle_120
            temp_obs[133:133+3] = _rvel_angle_240
            temp_obs[136:136+3] = _rvel_angle_0

            # finger tip force
            _f_force_angle_0 = copy(obs[90])
            _f_force_angle_120 = copy(obs[91])
            _f_force_angle_240 = copy(obs[92])
            temp_obs[90] = _f_force_angle_120
            temp_obs[91] = _f_force_angle_240
            temp_obs[92] = _f_force_angle_0

            # finger tip pos and vel
            fingertip_position, fingertip_velocity = env.sim_env.platform.forward_kinematics(
                temp_obs[111:111+9], temp_obs[130:130+9]
            )
            fingertip_position = np.array(fingertip_position, dtype=np.float32)
            fingertip_velocity = np.array(fingertip_velocity, dtype=np.float32)
            temp_obs[93:93+9] = fingertip_position.reshape(9,)
            temp_obs[102:102+9] = fingertip_velocity.reshape(9,)

            raw_dataset['observations'][idx] = temp_obs

            # action
            _action_0 = copy(raw_dataset['actions'][idx][0:0+3])
            _action_120 = copy(raw_dataset['actions'][idx][3:3+3])
            _action_240 = copy(raw_dataset['actions'][idx][6:6+3])
            raw_dataset['actions'][idx][0:0+3] = _action_120
            raw_dataset['actions'][idx][3:3+3] = _action_240
            raw_dataset['actions'][idx][6:6+3] = _action_0

            if idx % 100000 == 0:
                pbar.set_postfix()
                pbar.update(100000)

    np.save(
        save_path, raw_dataset)
    print('Scuuessfuly saved')
    
    del raw_dataset
    gc.collect()


def symmetry_push_aug_cw120(raw_dataset, save_path):
    env = gym.make(
        "trifinger-cube-push-sim-expert-v0",
    )
    env.reset()
    angle = math.radians(-120)
    obs = []

    with tqdm(total=raw_dataset['observations'].shape[0], desc="Augmenting the push task cw120") as pbar:

        for idx, obs in enumerate(raw_dataset['observations']):

            temp_obs = copy(obs)
            # observations
            # ag
            temp_obs[0:0+2] = rot_augment(obs[0:0+2], angle)

            # act
            _act_angle_0 = copy(obs[3:3+3])
            _act_angle_120 = copy(obs[6:6+3])
            _act_angle_240 = copy(obs[9:9+3])
            temp_obs[3:3+3] = _act_angle_240
            temp_obs[6:6+3] = _act_angle_0
            temp_obs[9:9+3] = _act_angle_120

            # g
            temp_obs[12:12+2] = rot_augment(obs[12:12+2], angle)

            # confidence not change 15
            # delay not change 16

            # ag2
            temp_obs[17:17+2] = rot_augment(obs[17:17+2], angle)
            temp_obs[20:20+2] = rot_augment(obs[20:20+2], angle)
            temp_obs[23:23+2] = rot_augment(obs[23:23+2], angle)
            temp_obs[26:26+2] = rot_augment(obs[26:26+2], angle)
            temp_obs[29:29+2] = rot_augment(obs[29:29+2], angle)
            temp_obs[32:32+2] = rot_augment(obs[32:32+2], angle)
            temp_obs[35:35+2] = rot_augment(obs[35:35+2], angle)
            temp_obs[38:38+2] = rot_augment(obs[38:38+2], angle)

            #orientation and position
            ag_key_points = copy(temp_obs[17:17+24]).reshape(8, 3)
            ag_pos = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[
                0]
            ag_ori = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[
                1]
            temp_obs[41:41+4] = ag_ori
            temp_obs[45:45+3] = ag_pos

            # rob pos
            _rpos_angle_0 = copy(obs[69:69+3])
            _rpos_angle_120 = copy(obs[72:72+3])
            _rpos_angle_240 = copy(obs[75:75+3])
            temp_obs[69:69+3] = _rpos_angle_240
            temp_obs[72:72+3] = _rpos_angle_0
            temp_obs[75:75+3] = _rpos_angle_120

            # rob id not change 120

            # rob torque
            _rtorque_angle_0 = copy(obs[79:79+3])
            _rtorque_angle_120 = copy(obs[82:82+3])
            _rtorque_angle_240 = copy(obs[85:85+3])
            temp_obs[79:79+3] = _rtorque_angle_240
            temp_obs[82:82+3] = _rtorque_angle_0
            temp_obs[85:85+3] = _rtorque_angle_120

            # rob vel
            _rvel_angle_0 = copy(obs[88:88+3])
            _rvel_angle_120 = copy(obs[91:91+3])
            _rvel_angle_240 = copy(obs[94:94+3])
            temp_obs[88:88+3] = _rvel_angle_240
            temp_obs[91:91+3] = _rvel_angle_0
            temp_obs[94:94+3] = _rvel_angle_120

            # finger tip force
            _f_force_angle_0 = copy(obs[48])
            _f_force_angle_120 = copy(obs[49])
            _f_force_angle_240 = copy(obs[50])
            temp_obs[48] = _f_force_angle_240
            temp_obs[49] = _f_force_angle_0
            temp_obs[50] = _f_force_angle_120

            # finger tip pos and vel
            fingertip_position, fingertip_velocity = env.sim_env.platform.forward_kinematics(
                temp_obs[69:69+9], temp_obs[88:88+9]
            )
            fingertip_position = np.array(fingertip_position, dtype=np.float32)
            fingertip_velocity = np.array(fingertip_velocity, dtype=np.float32)
            temp_obs[51:51+9] = fingertip_position.reshape(9,)
            temp_obs[60:60+9] = fingertip_velocity.reshape(9,)

            raw_dataset['observations'][idx] = temp_obs

            # action
            _action_0 = copy(raw_dataset['actions'][idx][0:0+3])
            _action_120 = copy(raw_dataset['actions'][idx][3:3+3])
            _action_240 = copy(raw_dataset['actions'][idx][6:6+3])
            raw_dataset['actions'][idx][0:0+3] = _action_240
            raw_dataset['actions'][idx][3:3+3] = _action_0
            raw_dataset['actions'][idx][6:6+3] = _action_120

            if idx % 100000 == 0:
                pbar.set_postfix()
                pbar.update(100000)
    np.save(
        save_path, raw_dataset)
    print('Scuuessfuly saved')
    
    del raw_dataset
    gc.collect()


def symmetry_push_aug_ccw120(raw_dataset, save_path):
    env = gym.make(
        "trifinger-cube-push-sim-expert-v0",
    )
    env.reset()
    angle = math.radians(-240)
    obs = []

    with tqdm(total=raw_dataset['observations'].shape[0], desc="Augmenting the push task ccw120") as pbar:
        for idx, obs in enumerate(raw_dataset['observations']):

            temp_obs = copy(obs)
            # observations
            # ag
            temp_obs[0:0+2] = rot_augment(obs[0:0+2], angle)

            # act
            _act_angle_0 = copy(obs[3:3+3])
            _act_angle_120 = copy(obs[6:6+3])
            _act_angle_240 = copy(obs[9:9+3])
            temp_obs[3:3+3] = _act_angle_120
            temp_obs[6:6+3] = _act_angle_240
            temp_obs[9:9+3] = _act_angle_0

            # g
            temp_obs[12:12+2] = rot_augment(obs[12:12+2], angle)

            # confidence not change 15
            # delay not change 16

            # ag2
            temp_obs[17:17+2] = rot_augment(obs[17:17+2], angle)
            temp_obs[20:20+2] = rot_augment(obs[20:20+2], angle)
            temp_obs[23:23+2] = rot_augment(obs[23:23+2], angle)
            temp_obs[26:26+2] = rot_augment(obs[26:26+2], angle)
            temp_obs[29:29+2] = rot_augment(obs[29:29+2], angle)
            temp_obs[32:32+2] = rot_augment(obs[32:32+2], angle)
            temp_obs[35:35+2] = rot_augment(obs[35:35+2], angle)
            temp_obs[38:38+2] = rot_augment(obs[38:38+2], angle)

            #orientation and position
            ag_key_points = copy(temp_obs[17:17+24]).reshape(8, 3)
            ag_pos = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[
                0]
            ag_ori = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[
                1]
            temp_obs[41:41+4] = ag_ori
            temp_obs[45:45+3] = ag_pos

            # rob pos
            _rpos_angle_0 = copy(obs[69:69+3])
            _rpos_angle_120 = copy(obs[72:72+3])
            _rpos_angle_240 = copy(obs[75:75+3])
            temp_obs[69:69+3] = _rpos_angle_120
            temp_obs[72:72+3] = _rpos_angle_240
            temp_obs[75:75+3] = _rpos_angle_0

            # rob id not change 120

            # rob torque
            _rtorque_angle_0 = copy(obs[79:79+3])
            _rtorque_angle_120 = copy(obs[82:82+3])
            _rtorque_angle_240 = copy(obs[85:85+3])
            temp_obs[79:79+3] = _rtorque_angle_120
            temp_obs[82:82+3] = _rtorque_angle_240
            temp_obs[85:85+3] = _rtorque_angle_0

            # rob vel
            _rvel_angle_0 = copy(obs[88:88+3])
            _rvel_angle_120 = copy(obs[91:91+3])
            _rvel_angle_240 = copy(obs[94:94+3])
            temp_obs[88:88+3] = _rvel_angle_120
            temp_obs[91:91+3] = _rvel_angle_240
            temp_obs[94:94+3] = _rvel_angle_0

            # finger tip force
            _f_force_angle_0 = copy(obs[48])
            _f_force_angle_120 = copy(obs[49])
            _f_force_angle_240 = copy(obs[50])
            temp_obs[48] = _f_force_angle_120
            temp_obs[49] = _f_force_angle_240
            temp_obs[50] = _f_force_angle_0

            # finger tip pos and vel
            fingertip_position, fingertip_velocity = env.sim_env.platform.forward_kinematics(
                temp_obs[69:69+9], temp_obs[88:88+9]
            )
            fingertip_position = np.array(fingertip_position, dtype=np.float32)
            fingertip_velocity = np.array(fingertip_velocity, dtype=np.float32)
            temp_obs[51:51+9] = fingertip_position.reshape(9,)
            temp_obs[60:60+9] = fingertip_velocity.reshape(9,)

            raw_dataset['observations'][idx] = temp_obs

            # action
            _action_0 = copy(raw_dataset['actions'][idx][0:0+3])
            _action_120 = copy(raw_dataset['actions'][idx][3:3+3])
            _action_240 = copy(raw_dataset['actions'][idx][6:6+3])
            raw_dataset['actions'][idx][0:0+3] = _action_120
            raw_dataset['actions'][idx][3:3+3] = _action_240
            raw_dataset['actions'][idx][6:6+3] = _action_0

            if idx % 100000 == 0:
                pbar.set_postfix()
                pbar.update(100000)

    np.save(
        save_path, raw_dataset)
    print('Scuuessfuly saved')
    
    del raw_dataset
    gc.collect()
