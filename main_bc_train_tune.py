#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: qiang
"""

import argparse
import rrc_dataset_handler
import dataset_filter
import dataset_aug
import trainer
from d3rlpy.dataset import MDPDataset
import utils
import gc

import warnings


def main_bc_train_tune(args):
    args = utils.device_handler(args)
    args = utils.directory_handler(args)
    args = utils.rrc_task_handler(args)
    
    if args.norm and args.task_type =='push':
        warnings.warn('The current implementation only involves std normalizer and one value in the push task stays static(z coordinate of the cube); hence it cannot be normalized by std. So setting the norm hyperparameters in args to 0')
        args.norm = 0
        
    if not args.require_dataset_process:
        assert args.aug_dataset_path and args.tune_datase_path, "If you don't need to process the dataset, you must give the path of the aug dataset and tune dataset"
    else:
        if args.diff == 'mixed':
            args.aug_dataset_path = f'./save/{args.task}/datasets/turn_final_positive_aug.npy'
            args.tune_datase_path = f'./save/{args.task}/datasets/turn_final_positive.npy'
        elif args.diff == 'expert':
            args.aug_dataset_path = f'./save/{args.task}/datasets/positive_aug.npy'
            args.tune_datase_path = f'./save/{args.task}/datasets/positive.npy'

    rrc_dataset_handler.train_tune_dataset_process(
        args.aug_dataset_path, args.tune_datase_path)

    bc_impl = trainer.BehaviourCloning(args)
    train_dataset = MDPDataset.load(
        f'{args.save_path}/datasets/train_aug_dataset.h5')
    bc_impl.reset(args.bc_train_learning_rate)
    bc_impl.train(train_dataset,
                  args,
                  'train',
                  norm_params_path=f'{args.save_path}/datasets/train_aug_norm_params.npy')

    del train_dataset
    gc.collect

    tune_dataset = MDPDataset.load(
        f'{args.save_path}/datasets/tune_dataset.h5')
    bc_impl.reset(args.bc_tune_learning_rate)
    bc_impl.train(tune_dataset,
                  args,
                  'tune',
                  norm_params_path=f'{args.save_path}/datasets/train_aug_norm_params.npy')
    print('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', default=123, help='The experiment name')
    parser.add_argument('--print-progress', type=int,default=1, help='If print the logs or not')
    parser.add_argument('--use-gpu', type=int, default=1,help='If using the GPU for training')
    parser.add_argument('--task', type=str,default='real_lift_mix', help='The name of the task')
    parser.add_argument('--seed', type=int, default=0, help='The random seed value')
    
    parser.add_argument('--norm', default=1, help='If normlize the observation or not')
    parser.add_argument('--normby', default='params')
    parser.add_argument('--bc-train-epochs', type=int, default=50,help='How many epochs to train the BC in the augmented dataset')
    parser.add_argument('--bc-tune-epochs', type=int, default=50,help='How many epochs to tune the BC')
    parser.add_argument('--bc-batch-size', type=int,default=1024, help='Learning rate value')
    parser.add_argument('--bc-train-learning-rate', type=float,default=0.001, help='Learning rate value')
    parser.add_argument('--bc-tune-learning-rate', type=float,default=0.0008, help='Learning rate value')
    
    parser.add_argument('--save-path', default=None,help='The path for saving the dataset')
    parser.add_argument('--require-dataset-process', default=True)
    parser.add_argument('--aug-dataset-path', default=None)
    parser.add_argument('--tune-datase-path', default=None)
    parser.add_argument('--model-save-interval', default=1)
    
    args = parser.parse_args()
    main_bc_train_tune(args)
