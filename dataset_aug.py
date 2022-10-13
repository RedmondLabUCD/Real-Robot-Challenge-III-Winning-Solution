#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:36:50 2022

@author: qiang
"""
import rrc_dataset_handler
import gc
import numpy as np
import argparse
import os
import utils


def main_augment(args):
    args = utils.directory_handler(args)
    args = utils.rrc_task_handler(args)
    
    
    if not args.raw_dataset_path and args.diff == 'mixed':
        default_path = f'./save/{args.task}/datasets/turn_final_positive.npy'
        if os.path.exists(default_path):
            args.raw_dataset_path = default_path
        else:
            raise RuntimeError('You must filter the dataset if you do not have an exist filtered dataset')
    
    if args.diff == 'mixed':
        temp_p = os.path.split(args.raw_dataset_path)
        cw_save_path = temp_p[0] + "/" + temp_p[1].split('.')[0] + '_cw.npy'
        ccw_save_path = temp_p[0] + "/" + temp_p[1].split('.')[0] + '_ccw.npy'
        aug_save_path = temp_p[0] + "/" + temp_p[1].split('.')[0] + '_aug.npy'
        if args.task_type == 'push':
            for i in [0, 1]:
                dataset = rrc_dataset_handler.load_rrc_raw_dataset(args)
                if i == 0:
                    rrc_dataset_handler.symmetry_push_aug_cw120(dataset, cw_save_path)
                elif i == 1:
                    rrc_dataset_handler.symmetry_push_aug_ccw120(dataset, ccw_save_path)
                del dataset
                gc.collect()
        elif args.task_type == 'lift':
            for i in [0, 1]:
                dataset = rrc_dataset_handler.load_rrc_raw_dataset(args)
                if i == 0:
                    rrc_dataset_handler.symmetry_lift_aug_cw120(dataset, cw_save_path)
                elif i == 1:
                    rrc_dataset_handler.symmetry_lift_aug_ccw120(dataset, ccw_save_path)
                del dataset
                gc.collect()
        rrc_dataset_handler.concatenate_datasets(datasets_list=[args.raw_dataset_path,
                                                                cw_save_path,
                                                                ccw_save_path], save_path=aug_save_path)
    
    elif args.diff == 'expert':
        save_path = './save/{args.task}/datasets/positive.npy'
        cw_save_path = './save/{args.task}/datasets/positive_cw120.npy'
        ccw_save_path = './save/{args.task}/datasets/positive_ccw120.npy'
        if args.task_type == 'push':
            for i in [0, 1]:
                dataset = rrc_dataset_handler.load_rrc_raw_dataset(args)
                if i == 0:
                    np.save(save_path, dataset)
                    rrc_dataset_handler.symmetry_push_aug_cw120(dataset, cw_save_path)
                elif i == 1:
                    rrc_dataset_handler.symmetry_push_aug_ccw120(dataset, ccw_save_path)
                del dataset
                gc.collect()
        elif args.task_type == 'lift':
            for i in [0, 1]:
                dataset = rrc_dataset_handler.load_rrc_raw_dataset(args)
                if i == 0:
                    np.save(save_path, dataset)
                    rrc_dataset_handler.symmetry_lift_aug_cw120(dataset, cw_save_path)
                elif i == 1:
                    rrc_dataset_handler.symmetry_lift_aug_ccw120(dataset, ccw_save_path)
                del dataset
                gc.collect()
        rrc_dataset_handler.concatenate_datasets(datasets_list=[save_path,
                                                                cw_save_path,
                                                                ccw_save_path])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='real_lift_mix', help='The name of the task')
    parser.add_argument('--raw-dataset-path', default=None, help='If load the exist dataset')
    parser.add_argument('--save-path', default=None, help='Where to save the models and data')
    args = parser.parse_args()
    main_augment(args)
