import torch
import numpy as np
import pandas as pd

import math
import time
import os
import json
import warnings
import argparse

from models.dynamic_net import Vcnet, TR
from data.data import *
from utils.eval import *

warnings.filterwarnings("ignore", category=UserWarning)

#def save_checkpoint(state, delta, model_name='', checkpoint_dir='.'):
#    filename = os.path.join(checkpoint_dir, model_name + f'_delta_{delta}' + '_ckpt.pth.tar')
#    print('=> Saving checkpoint to {}'.format(filename))
#    torch.save(state, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/simu3/eval', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/simu3/eval', help='dir to save result')

    # common
    parser.add_argument('--num_dataset', type=int, default=50, help='num of datasets to train')

    # training
    parser.add_argument('--n_epochs', type=int, default=800, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=bool, default=False, help='print train info freq or not') # if False, verbose_num does not matter
    parser.add_argument('--verbose_num', type=int, default=200, help='number of epochs to print train info')

    # significance level
    parser.add_argument('--alpha', type=float, default=0.05, help='significance level for test')

    args = parser.parse_args()

    # six scenario
    #delta_list = [x/10 for x in range(0, 6, 1)]
    delta_list = [0, 0.3, 0.5 ]#, 1] #[0, 0.5] #[x/10 for x in range(0, 6, 1)]

    # splitting ratio, inf_ratio; noise size, rho
    inf_ratio = 0.3 #0.2 #0.1 #0.15 #0.08 #0.15 #0.3
    rho = 0.1 #0.2 #0.45 #0.5 #0.4 #0.3 #0.15 #0.135 #0.12 #0.1 0.05, 0.08 too small for ratio = 0.08 #0.15 #0.4

    # data
    load_path = args.data_dir
    num_dataset = args.num_dataset

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # fixed parameter for optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 5e-3

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose_num

    cfg_density = [(6, 50, 1, 'relu'), (50, 50, 1, 'relu')]
    num_grid = 10
    cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
    degree = 2
    knots = [0.33, 0.66]
    model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
    model._initialize_weights()

    tr_knots = list(np.arange(0.1, 1, 0.1))
    tr_degree = 2
    TargetReg = TR(tr_degree, tr_knots)
    TargetReg._initialize_weights()

    init_lr = 0.0001
    alpha = 0.5
    tr_init_lr = 0.001
    beta = 1.

    for delta in delta_list:
        print(f'Start the case for delta = {delta}')

        p_val = np.zeros(num_dataset)
        run_time = np.zeros(num_dataset)
        for _ in range(num_dataset):
            print(f'dataset: {_ + 1}/{num_dataset}')
            #cur_save_path = save_path + '/' + str(_)
            #if not os.path.exists(cur_save_path):
            #    os.makedirs(cur_save_path)

            data = pd.read_csv(load_path + '/' + str(_) + f'/delta_{delta}_data.txt', header=None, sep=' ')
            data = data.to_numpy()
            t_grid_dat = pd.read_csv(load_path + '/' + str(_) + f'/delta_{delta}_t_grid.txt', header=None, sep=' ')
            t_grid_dat = t_grid_dat.to_numpy()

            train_matrix, test_matrix, t_grid = split(data, t_grid_dat, inf_ratio)

            # train_matrix, test_matrix, t_grid = simu_data1(500, 200)
            train_loader = get_iter(train_matrix, batch_size=train_matrix.shape[0], shuffle=True)
            test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

            # reinitialize model
            model._initialize_weights()
            TargetReg._initialize_weights()

            # define optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)
            tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

            # get the start time
            st = time.time()

            for epoch in range(num_epoch):
                for idx, (inputs, y) in enumerate(train_loader):
                    t = inputs[:, 0]
                    x = inputs[:, 1:]

                    optimizer.zero_grad()
                    out = model.forward(t, x)
                    trg = TargetReg(t)
                    loss = criterion(out, y, alpha=alpha) + criterion_TR(out, trg, y, beta=beta)
                    loss.backward()
                    optimizer.step()

                    tr_optimizer.zero_grad()
                    out = model.forward(t, x)
                    trg = TargetReg(t)
                    tr_loss = criterion_TR(out, trg, y, beta=beta)
                    tr_loss.backward()
                    tr_optimizer.step()

                if args.verbose == True:
                    if epoch % verbose == 0:
                        print('current epoch: ', epoch)
                        print('loss: ', loss.data)

            #t_grid_hat, mse = curve(model, test_matrix, t_grid, targetreg=TargetReg)

            #mse = float(mse)
            #print('current loss: ', float(loss.data))
            #print('current test loss: ', mse)
            #print('-----------------------------------------------------------------')
            #save_checkpoint({
            #    'best_test_loss': mse,
            #    'model_state_dict': model.state_dict(),
            #    'TR_state_dict': TargetReg.state_dict(),
            #}, delta=delta, checkpoint_dir=cur_save_path)
            #print('-----------------------------------------------------------------')

            #p_val0 = test_given_ratio(model, test_matrix, t_grid_hat, rho, TargetReg)
            Delta = calculate_delta0(model, test_matrix, targetreg = TargetReg)
            p_val0 = test_from_delta(Delta, rho)

            # get the end time
            et = time.time()
            # get the execution time
            elapsed_time = et - st

            run_time[_] = elapsed_time
            #print('p_value: ', p_val0)
            p_val[_] = p_val0

        print(f'End the case for delta = {delta}')
        data_file = os.path.join(save_path, f'p_val_one_split_delta_{delta}_at_inf_ratio_{inf_ratio}_rho_{rho}_num_dataset_{num_dataset}.txt')
        np.savetxt(data_file, p_val)
        print('p-values saved to', save_path)
        time_file = os.path.join(save_path, f'run_time_one_split_delta_{delta}_at_inf_ratio_{inf_ratio}_rho_{rho}_num_dataset_{num_dataset}.txt')
        np.savetxt(time_file, run_time)
        print('run time saved to', save_path)
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')

    print('Complete all p-values; now calculate the rejection rate...')
    rej_rate = np.zeros(len(delta_list))
    time_cost = np.zeros(len(delta_list))
    for _ in range(len(delta_list)):
        delta = delta_list[_]
        try:
            p_val = pd.read_csv(save_path + f'/p_val_one_split_delta_{delta}_at_inf_ratio_{inf_ratio}_rho_{rho}_num_dataset_{num_dataset}.txt', header = None)
            run_time = pd.read_csv(save_path + f'/run_time_one_split_delta_{delta}_at_inf_ratio_{inf_ratio}_rho_{rho}_num_dataset_{num_dataset}.txt', header = None)
        except FileNotFoundError:
            continue
        p_val = p_val.to_numpy()
        rej_rate[_] = (p_val < args.alpha).mean()

        run_time = run_time.to_numpy()
        time_cost[_] = run_time.mean()

    data_file = os.path.join(save_path, f'rej_rate_one_split_at_inf_ratio_{inf_ratio}_rho_{rho}_num_dataset_{num_dataset}.txt')
    np.savetxt(data_file, np.array([delta_list, rej_rate, time_cost]))
    print('rejection rate done, saved to', save_path)
    print('Program done')
