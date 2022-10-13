#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: qiang
"""
import models
import utils
import torch
from rrc_dataset_handler import torch_loader
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import os
import gc
import random


def set_seed(seed):
    utils.set_seed(seed)
    models.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BehaviourCloning():
    def __init__(self,
                 args):
        self.args = args

        if args.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('I am using CUDA')
        else:
            self.device = torch.device('cpu')

        self.model = models.BcNet(obs_dim=args.obs_dim,
                                  action_dim=args.action_dim,
                                  bias=True
                                  ).to(self.device)
        
        self.criterion = torch.nn.MSELoss().to(self.device)
        
        self.normlizor = utils.Normlizor()
        self.train_iterator = None
        

    def reset(self, lr):
        if self.train_iterator:
            del self.train_iterator
            gc.collect()

    def train(self, dataset, args, mode, norm_params_path=None, d4rl_dataset_path=None):
        if mode == 'train':
            epochs = args.bc_train_epochs
            lr = args.bc_train_learning_rate
        elif mode == 'tune':
            epochs = args.bc_tune_epochs
            lr = args.bc_tune_learning_rate
            
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr,
                                          weight_decay=0,
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          amsgrad=False
                                          )
        
        self.train_iterator = utils.gen_iterator(
            dataset, batch_size=args.bc_batch_size)
        if self.args.norm:
            if self.args.normby == 'params':
                if norm_params_path:
                    self.normlizor.init_norm_with_params(
                        norm_params_path, 'std')
            elif self.args.normby == 'dataset':
                self.normlizor.init_norm_with_params(d4rl_dataset_path, 'std')

        train_logger = utils.TrainLogger(
            experiment_name=f'{args.exp_name}_{mode}',
            save_metrics=True,
            root_dir=f'./save/{args.task}/models',
            verbose=True,
            tensorboard_dir=1,
            with_timestamp=True,
        )

        total_step = 0
        for epoch in range(1, epochs + 1):
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(len(self.train_iterator)),
                disable=not self.args.print_progress,
                desc=f"Epoch {int(epoch)}/{args.bc_train_epochs}",
            )

            self.train_iterator.reset()
            self.model.train()
            for itr in range_gen:

                with train_logger.measure_time("step"):
                    # pick transitions
                    with train_logger.measure_time("sample_batch"):
                        batch = next(self.train_iterator)

                    # update parameters
                    with train_logger.measure_time("algorithm_update"):
                        if self.args.norm:
                            opt = self.model(torch.Tensor(self.normlizor.batch_norm(
                                batch.observations)).to(torch.float32).to(self.device))
                        else:
                            opt = self.model(torch.Tensor(batch.observations).
                                             to(torch.float32).to(self.device))
                        self.optimizer.zero_grad()
                        loss = self.criterion(opt, torch.Tensor(
                            batch.actions).to(torch.float32).to(self.device))
                        loss.backward()
                        self.optimizer.step()

                    # record metrics
                    train_logger.add_metric(
                        'loss', loss.cpu().detach().numpy())
                    epoch_loss['loss'].append(loss.cpu().detach().numpy())

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                    total_step += 1

            train_logger.commit(epoch, total_step)

            # save model parameters
            if epoch % self.args.model_save_interval == 0:
                self.save(
                    epoch, f'./save/{args.task}/models/{train_logger._experiment_name}')

    def select_action(self, obs, norm=True):
        if self.args.norm or norm:
            obs = self.normlizor.norm(obs)
        return self.model(torch.Tensor([obs]).to(torch.float32).to(self.device)).cpu().detach().numpy()[0]

    def save(self, epoch, path=None):
        torch.save(self.model.state_dict(), f'{path}/ckpt_{epoch}.pth')

    def load(self, path=None):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def train_contra_net(args, turn_num):
    set_seed(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')

    model = models.ContraNet(obs_dim=args.obs_dim,
                             action_dim=args.action_dim,
                             bias=True
                             ).to(device)
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(
    ), lr=args.filter_learning_rate, weight_decay=3e-3, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    train_loader = torch_loader(filtered_dataset=f'{args.save_path}/datasets/turn{turn_num}_positive.npy',
                                raw_dataset=f'{args.save_path}/datasets/turn{turn_num}_negative.npy',
                                train_batch_size=args.filter_batch_size,
                                shuffle=True)
    for epoch in range(args.filter_train_epochs):
        epoch_loss = defaultdict(list)
        model.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {int(epoch)}/ {int(args.filter_train_epochs)}") as pbar:
            for idx, [observations, actions, labels] in enumerate(train_loader):

                if torch.cuda.is_available() and args.use_gpu:
                    observations = observations.cuda(
                        non_blocking=True).to(torch.float32)
                    actions = actions.cuda(
                        non_blocking=True).to(torch.float32)
                    labels = labels.cuda(
                        non_blocking=True).to(torch.float32)
                else:
                    observations = torch.tensor(observations).to(
                        torch.float32).to(device)
                    actions = torch.tensor(actions).to(
                        torch.float32).to(device)
                    labels = torch.tensor(labels).to(
                        torch.float32).to(device)

                pred = model(observations, actions)
                loss = criterion(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss['loss'].append(loss.cpu().detach().numpy())

                if idx % 10 == 0:
                    mean_loss = {
                        k: np.mean(v) for k, v in epoch_loss.items()
                    }
                    pbar.set_postfix(mean_loss)
                    pbar.update(10)
    torch.save(model.state_dict(),
               f'{args.save_path}/models/contra_filter_turn{turn_num}.pth')
    del train_loader, criterion, model
    gc.collect()


'''
Using this class requires more memory, we would suggest using this with the memory
of at least 32GB, but using this is easier.
'''
class ContrastiveFilter():
    def __init__(self,
                 raw_dataset,
                 task_type,
                 args):

        self.args = args
        if task_type == 'push':
            obs_dim = utils.PUSH_TASK_OBS_DIM
        elif task_type == 'lift':
            obs_dim = utils.LIFT_TASK_OBS_DIM
        else:
            raise RuntimeError(
                'The input task type is invalid')

        if args.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('I am using CUDA')
        else:
            self.device = torch.device('cpu')

        self.model = models.ContraNet(obs_dim=obs_dim,
                                      action_dim=utils.ACTION_DIM,
                                      bias=True
                                      ).to(self.device)

        self.criterion = torch.nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=args.lr,
                                          weight_decay=3e-3,
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          amsgrad=False
                                          )
        self.raw_dataset = raw_dataset

    def train(self, pos_dataset, neg_dataset, batch_size, epochs, exp_name):
        self.train_loader = torch_loader(filtered_dataset=pos_dataset,
                                         raw_dataset=neg_dataset,
                                         train_batch_size=batch_size,
                                         shuffle=True)

        train_logger = utils.TrainLogger(
            experiment_name=exp_name,
            save_metrics=True,
            root_dir='./save/trained_models',
            verbose=True,
            tensorboard_dir=1,
            with_timestamp=True,
        )

        self.model.train()
        epoch_loss = defaultdict(list)
        total_step = 0
        for epoch in range(1, epochs + 1):
            epoch_loss = defaultdict(list)

            with tqdm(total=len(self.train_loader), desc=f"Epoch {int(epoch)}/ {int(epochs+1)}") as pbar:
                for idx, [observations, actions, labels] in enumerate(self.train_loader):

                    if torch.cuda.is_available():
                        observations = observations.cuda(
                            non_blocking=True).to(torch.float32)
                        actions = actions.cuda(
                            non_blocking=True).to(torch.float32)
                        labels = labels.cuda(
                            non_blocking=True).to(torch.float32)

                    else:
                        observations = torch.tensor(observations).to(
                            torch.float32).to(self.device)
                        actions = torch.tensor(actions).to(
                            torch.float32).to(self.device)
                        labels = torch.tensor(labels).to(
                            torch.float32).to(self.device)

                    pred = self.model(observations, actions)
                    loss = self.criterion(pred, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    train_logger.add_metric(
                        'loss', loss.cpu().detach().numpy())
                    epoch_loss['loss'].append(loss.cpu().detach().numpy())

                    if idx % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        pbar.set_postfix(mean_loss)
                        pbar.update(10)
                    total_step += 1

                train_logger.commit(epoch, total_step)

            if epoch % self.args.model_save_freq == 0:
                self.save(
                    epoch, path=f'./save/trained_models/{train_logger._experiment_name}')

    def filterit(self, turn, prob_th):
        overall_length = self.raw_dataset['timeouts'].shape[0]

        temp_obs = []
        temp_actions = []
        first_flag = True
        temp_first_flag = True
        temp_obs_numpy = None
        temp_actions_numpy = None
        seq_num = 0
        indexes = []
        index_count = 0
        with torch.no_grad():
            self.model.eval()
            for idx, timeout in enumerate(self.raw_dataset['timeouts']):
                temp_obs.append(self.raw_dataset['observations'][idx].tolist())
                temp_actions.append(self.raw_dataset['actions'][idx].tolist())

                if timeout:
                    if torch.cuda.is_available():
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                        temp_actions_tensor = torch.tensor(temp_actions.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                    else:
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).to(
                            torch.float32).to(self.device)
                        temp_actions_tensor = torch.tensor(
                            temp_actions.copy()).to(torch.float32).to(self.device)

                    prob = self.model(temp_obs_tensor, temp_actions_tensor).cpu(
                    ).detach().numpy()[..., 0].sum()

                    if prob >= prob_th:
                        if first_flag:
                            obs = np.array(temp_obs)
                            actions = np.array(temp_actions)
                            seq_num += 1
                            first_flag = False
                        else:
                            if seq_num % 100 == 0:
                                obs = np.concatenate((obs, temp_obs_numpy))
                                actions = np.concatenate(
                                    (actions, temp_actions_numpy))
                                temp_obs_numpy = np.array(temp_obs)
                                temp_actions_numpy = np.array(temp_actions)
                                seq_num += 1
                            else:
                                if temp_first_flag:
                                    temp_obs_numpy = np.array(temp_obs)
                                    temp_actions_numpy = np.array(temp_actions)
                                    temp_first_flag = False
                                temp_obs_numpy = np.concatenate(
                                    (temp_obs_numpy, np.array(temp_obs)))
                                temp_actions_numpy = np.concatenate(
                                    (temp_actions_numpy, np.array(temp_actions)))
                                seq_num += 1
                        indexes.append(index_count)
                    temp_obs = []
                    temp_actions = []
                    index_count += 1

                if self.args.print_progress:
                    if idx % 5e5 == 0:
                        print(f'Positive progress: {idx} / {overall_length}')
            obs = np.concatenate((obs, temp_obs_numpy))
            actions = np.concatenate((actions, temp_actions_numpy))
        pos_temp_dataset = {}
        pos_temp_dataset['observations'] = obs
        pos_temp_dataset['actions'] = actions

        temp_obs = []
        temp_actions = []
        first_flag = True
        temp_first_flag = True
        temp_obs_numpy = None
        temp_actions_numpy = None
        seq_num = 0
        with torch.no_grad():
            self.model.eval()
            for idx, timeout in enumerate(self.raw_dataset['timeouts']):
                temp_obs.append(self.raw_dataset['observations'][idx].tolist())
                temp_actions.append(self.raw_dataset['actions'][idx].tolist())

                if timeout:
                    if torch.cuda.is_available():
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                        temp_actions_tensor = torch.tensor(temp_actions.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                    else:
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).to(
                            torch.float32).to(self.device)
                        temp_actions_tensor = torch.tensor(
                            temp_actions.copy()).to(torch.float32).to(self.device)

                    prob = self.model(temp_obs_tensor, temp_actions_tensor).cpu(
                    ).detach().numpy()[..., 0].sum()

                    if not prob >= prob_th:
                        if first_flag:
                            obs = np.array(temp_obs)
                            actions = np.array(temp_actions)
                            seq_num += 1
                            first_flag = False
                        else:
                            if seq_num % 100 == 0:
                                obs = np.concatenate((obs, temp_obs_numpy))
                                actions = np.concatenate(
                                    (actions, temp_actions_numpy))
                                temp_obs_numpy = np.array(temp_obs)
                                temp_actions_numpy = np.array(temp_actions)
                                seq_num += 1
                            else:
                                if temp_first_flag:
                                    temp_obs_numpy = np.array(temp_obs)
                                    temp_actions_numpy = np.array(temp_actions)
                                    temp_first_flag = False
                                temp_obs_numpy = np.concatenate(
                                    (temp_obs_numpy, np.array(temp_obs)))
                                temp_actions_numpy = np.concatenate(
                                    (temp_actions_numpy, np.array(temp_actions)))
                                seq_num += 1
                    temp_obs = []
                    temp_actions = []

                if self.args.print_progress_log:
                    if idx % 5e5 == 0:
                        print(f'Negative progress: {idx} / {overall_length}')

            obs = np.concatenate((obs, temp_obs_numpy))
            actions = np.concatenate((actions, temp_actions_numpy))

        neg_temp_dataset = {}
        neg_temp_dataset['observations'] = obs
        neg_temp_dataset['actions'] = actions

        if self.args.save_interm_dataset:
            if not os.path.exists(f'{self.args.save_folder}/dataset'):
                os.mkdir(f'{self.args.save_folder}/dataset')
            print(f'Saving the dataset filtered, turn {turn}')
            np.save(
                f'{self.args.save_folder}/dataset/DatasetFilteredTurn{turn}_POS.npy', pos_temp_dataset)
            np.save(
                f'{self.args.save_folder}/dataset/DatasetFilteredTurn{turn}_NEG.npy', neg_temp_dataset)

        return pos_temp_dataset, neg_temp_dataset, indexes

    def save(self, epoch, path=None):
        torch.save(self.model.state_dict(), f'{path}/ckpt_{epoch}.pth')

    def load(self, path=None):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
