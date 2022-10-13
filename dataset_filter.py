#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 22:00:42 2022

@author: qiang
"""
import argparse
import rrc_dataset_handler
import trainer
import gc
import torch
import models
import utils

def main_filter(args):
    if args.task != "real_push_mix" and args.task != "real_lift_mix":
        raise RuntimeError("Only the mixed dataset can be filtered")
    args = utils.device_handler(args)
    args = utils.directory_handler(args)
    args = utils.rrc_task_handler(args)
    
    if not args.exist_filter_path:
        print('Start to train the filter')
        for turn in range(args.turns):
            if turn == args.turns - 1:
                args.turn_num = "_final"
            else:
                args.turn_num = turn
            if turn == 0:
                raw_dataset = rrc_dataset_handler.load_rrc_raw_dataset(args)
                idxs = rrc_dataset_handler.filter_by_reward(raw_dataset, args)
                print(f'After turn {turn}, {len(idxs)} number of episodes are filtered out')
            else:
                raw_dataset = rrc_dataset_handler.load_rrc_raw_dataset(args)
                model = models.ContraNet(
                    obs_dim=args.obs_dim, action_dim=args.action_dim, bias=True).to(args.device)
                print('loading from:')
                print(f'{args.save_path}/models/contra_filter_turn{turn-1}.pth')
                model.load_state_dict(torch.load(
                    f'{args.save_path}/models/contra_filter_turn{turn-1}.pth'))
                with torch.no_grad():
                    model.eval()
                    idxs = rrc_dataset_handler.filter_by_net(raw_dataset, model, args.confs[turn-1], args)
                    print(f'After turn {turn}, {len(idxs)} number of episodes are filtered out')
            del raw_dataset
            gc.collect()
            if turn < args.turns-1:
                trainer.train_contra_net(args, turn)
                
    else:
        print('Using the exist filter for splitting the dataset')
        args.turn_num = "_final"
        raw_dataset = rrc_dataset_handler.load_rrc_raw_dataset(args)
        model = models.ContraNet(
            obs_dim=args.obs_dim, action_dim=args.action_dim, bias=True).to(args.device)
        print('loading from:')
        print(f'{args.exist_filter_path}')
        model.load_state_dict(torch.load(
            f'{args.exist_filter_path}'))
        with torch.no_grad():
            model.eval()
            rrc_dataset_handler.filter_by_net(raw_dataset, model, args.confs[-1], args)
        del raw_dataset
        gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--turns', default=None, help='How many turns to run the filter')
    parser.add_argument('--seed', default=0, help='The random seed value')
    parser.add_argument('--use-gpu', default=1, help='If using the GPU for training')
    parser.add_argument('--task', default='real_lift_mix', help='The name of the task')
    parser.add_argument('--filter-train-epochs', default=20, help='How many epochs to train the filter in each turn')
    parser.add_argument('--filter-batch-size', default=1024, help='Batch size of training the filter')
    parser.add_argument('--filter-learning-rate', default=0.001, help='Learning rate')
    parser.add_argument('--raw-dataset-path', default=None, help='If load the exist dataset')
    parser.add_argument('--save-path', default=None, help='Where to save the models and data')
    parser.add_argument('--exist-filter-path', default=None, help='If load the filter')
    args = parser.parse_args()
    main_filter(args)
