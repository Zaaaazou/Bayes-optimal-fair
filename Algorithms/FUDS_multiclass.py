import random
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader.dataloader import CustomDataset
from utils import cal_disparity,cal_acc,cal_t_bound, number_of_sample, cal_t_bound_multiclass, number_of_sample_multiclass, cal_acc_multiclass, cal_disparity_multiclass
import time

from DataLoader.dataloader import FairnessDataset
import torch.optim as optim
from models import Classifier


def get_level(dataset_name,fairness):
    if dataset_name == 'AdultCensus':
        if fairness == 'DP':
            level_list = np.arange(50)/250
        if fairness == 'EO':
            level_list = np.arange(50) / 1250
        if fairness == 'PE':
            level_list = np.arange(10) / 250
        if fairness == 'OAE':
            level_list = np.arange(10) / 2000

    if dataset_name == 'Synthetic_multiclass':
        if fairness == 'DP':
            level_list = np.arange(10)/250
        if fairness == 'EO':
            level_list = np.arange(50) /1250
        if fairness == 'PE':
            level_list = np.arange(50) /250
        if fairness == 'OAE':
            level_list = np.arange(10) / 2000

    if dataset_name == 'COMPAS':
        if fairness == 'DP':
            level_list = np.arange(10)/50
        if fairness == 'EO':
            level_list = np.arange(10) /50
        if fairness == 'PE':
            level_list = np.arange(10) /50
        if fairness == 'OAE':
            level_list = np.arange(10) / 500
    if dataset_name == 'Lawschool':
        if fairness == 'DP':
            level_list = np.arange(10)/250
        if fairness == 'EO':
            level_list = np.arange(10) /125
        if fairness == 'PE':
            level_list = np.arange(10) /2000
        if fairness == 'OAE':
            level_list = np.arange(10) / 200
    return level_list



def FUDS_multiclass(dataset, dataset_name, net, optimizer, lr_schedule, delta, device, n_epochs=200, batch_size=2048, seed=0):
    training_tensors, testing_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = training_tensors
    X_test, Y_test, Z_test, XZ_test = testing_tensors

    print(Z_train)

    unique_Z = Z_train.unique()
    unique_Y = Y_train.unique()

    # Dictionary to hold data subsets for each (Z, Y) combination and their current instances
    subsets = {}
    subset_sizes_ori = {}
    subsets_now = {}
    subset_sizes = {}
    subset_sizes_now = {}
    probabilities = {}
    n = len(X_train)


    for z in unique_Z:
        for y in unique_Y:
            print("ATTRIBUTS ET LABEL DE Y ET Z")
            print(z, y)
            mask = (Z_train == z) & (Y_train == y)
            key = f"{int(z)}{int(y)}"
            subsets[key] = XZ_train[mask]
            subsets_now[key] = subsets[key].clone().detach().numpy()
            subset_sizes_ori[key] = len(subsets[key])
            subset_sizes_now[key] = subset_sizes_ori[key]
            probabilities[key] = subset_sizes_ori[key] / len(XZ_train)
        
    tmin, tmax = cal_t_bound_multiclass(probabilities, unique_Z)
    tmid = 0

    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    if batch_size == 'full':
        batch_size_ = XZ_train.shape[0]
    elif isinstance(batch_size, int):
        batch_size_ = batch_size
    data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)
    loss_function = nn.BCELoss()

#####Pre-train the model######
    with tqdm(range(n_epochs//4)) as epochs:
        epochs.set_description(f"Pre-train the model:  dataset: {dataset_name}, seed: {seed}, level:{delta}")

        for epoch in epochs:
            net.train()
            for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
                xz_batch, y_batch, z_batch = xz_batch.to(device), y_batch.to(device), z_batch.to(device)
                Yhat = net(xz_batch)

                cost = loss_function(Yhat.squeeze(), y_batch)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                epochs.set_postfix(loss=cost.item())
            if dataset_name == 'AdultCensus':
                lr_schedule.step()


########Update the threshold parameter#######
    for T in range(15):
        loss_function = nn.BCELoss()

        with tqdm(range(n_epochs//20)) as epochs:
            epochs.set_description(f"training with dataset: {dataset_name}, seed: {seed}, level:{delta}, T: {T}")

            for epoch in epochs:
                net.train()
                for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
                    xz_batch, y_batch, z_batch = xz_batch.to(device), y_batch.to(device), z_batch.to(device)
                    y_batch = y_batch.unsqueeze(1)
                    z_batch = z_batch.unsqueeze(1)

                    Yhat = net(xz_batch)

                    cost = loss_function(Yhat, y_batch)

                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()

                    epochs.set_postfix(loss=cost.item())

                if dataset_name == 'AdultCensus':
                    lr_schedule.step()

    ########choose the model with best performance on validation set###########

        disparities = {}
        with torch.no_grad():
            Yhat_train = net(XZ_train).detach().squeeze().numpy()>0.5
            for z in unique_Z:
                for w in unique_Z:
                    if w != z:
                        disparities[f"{int(z)}{int(w)}"] = Yhat_train[Z_train == z].mean() - Yhat_train[Z_train == w].mean()
            disparity = 1/len(unique_Z) * sum([np.abs(disparities[f"{int(z)}{int(w)}"]) for z in unique_Z for w in unique_Z if w != z])
            print('VOICI LA REPONSE JE SUIS LA')
            print(disparity)
            print(delta)

            if disparity > delta:
                tmin = tmid
            elif disparity < -delta:
                tmax = tmid
            elif disparity > 0:
                if tmid > 0:
                    tmax = tmid
                else:
                    tmin = tmid
            else:
                if tmid > 0:
                    tmax = tmid
                else:
                    tmin = tmid

            tmid = (tmax + tmin) / 2

        subset_sizes = number_of_sample_multiclass(subset_sizes_ori, Z_train,  tmid, n)

        index = {}
        subsets_syn = {}
        Y_syn = {}
        Z_syn = {}
        subsets_conc = []
        Y_conc = []
        Z_conc = []

        for z in range(len(unique_Z)):
            key_0 = f"{int(z)}{int(0)}"
            key_1 = f"{int(z)}{int(1)}"
            if subset_sizes[key_1] > subset_sizes_now[key_1]:
                if subset_sizes[key_1] > (subset_sizes_now[key_1] + subset_sizes_ori[key_1]):
                    index_1 = np.random.choice(len(subsets[key_1]), subset_sizes[key_1] - subset_sizes_now[key_1], replace=True)
                    subsets_syn[key_1]= subsets[key_1][index_1, :]
                    subsets_now[key_1] = torch.concat([subsets_now[key_1], subsets_syn[key_1]])
                else:
                    index_1 = np.random.choice(len(subsets[key_1]), subset_sizes[key_1]-subset_sizes_now[key_1], replace = False)
                    subsets_syn[key_1] = subsets[key_1][index_1, :]
            else:
                index_1 = np.random.choice(len(subsets_now[key_1]), subset_sizes[key_1], replace = False)
                subsets_now = subsets_now[index_1, :]
            
            if subset_sizes[key_0] > subset_sizes_now[key_0]:
                if subset_sizes[key_0] > (subset_sizes_now[key_0] + subset_sizes_ori[key_0]):
                    index_0 = np.random.choice(len(subsets[key_0]), subset_sizes[key_0] - subset_sizes_now[key_0], replace=True)
                    subsets_syn[key_0]= subsets[key_0][index_0, :]
                    subsets_now[key_0] = torch.concat([subsets_now[key_0], subsets_syn[key_0]])
                else:
                    index_0 = np.random.choice(len(subsets[key_0]), subset_sizes[key_0]-subset_sizes_now[key_0], replace = False)
                    subsets_syn[key_0] = subsets[key_0][index_0, :]

            Y_syn[key_0] = torch.zeros(len(subsets_now[key_0]))
            Y_syn[key_1] = torch.ones(len(subsets_now[key_1]))
            Y_conc.append([Y_syn[key_1], Y_syn[key_0]])
            Z_syn[f"{int(z)}{int(0)}"] = int(z)*torch.ones(len(subsets_now[f"{int(z)}{int(0)}"]))
            Z_conc.append([Z_conc[key_1], Z_conc[key_0]])
            
            subsets_conc.append([subsets_syn[key_1], subsets_syn[key_0]])



        XZ_syn = torch.concat(subsets_conc)
        Y_syn = torch.concat(Y_conc)
        Z_syn = torch.concat(Z_conc)


        custom_dataset = CustomDataset(XZ_syn, Y_syn, Z_syn)
        if batch_size == 'full':
            batch_size_ = XZ_train.shape[0]
        elif isinstance(batch_size, int):
            batch_size_ = batch_size
        data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)

        for z in range(len(unique_Z)):
            key_0 = f"{int(z)}{int(0)}"
            key_1 = f"{int(z)}{int(1)}"
            subset_sizes_now[key_0] = subset_sizes[key_0]
            subset_sizes_now[key_1] = subset_sizes[key_1]


####Evaluate performance##########
    eta_test = net(XZ_test).detach().cpu().numpy().squeeze()


    Y_test_np = Y_test.clone().detach().numpy()
    Z_test_np = Z_test.clone().detach().numpy()

    t = [0.5 for z in unique_Z]

    acc = cal_acc_multiclass(eta_test, Y_test_np, Z_test_np, t, len(unique_Z))
    disparity = cal_disparity_multiclass(eta_test,Z_test_np,t, unique_Z)

    data = [seed,dataset_name,delta,acc, np.abs(disparity)]
    columns = ['seed','dataset','level','acc', 'disparity']
    df_test = pd.DataFrame([data], columns=columns)
    return df_test



def get_training_parameters(dataset_name):
    if dataset_name == 'AdultCensus':
        n_epochs = 200
        lr = 1e-1
        batch_size = 512

    if dataset_name == 'Synthetic_multiclass':
        n_epochs = 200
        lr = 1e-1
        batch_size = 512

    if dataset_name == 'COMPAS':
        n_epochs = 500
        lr = 5e-4
        batch_size = 2048

    if dataset_name == 'Lawschool':
        n_epochs = 200
        lr = 2e-4
        batch_size = 2048
    return n_epochs,lr,batch_size


def training_FUDS_multiclass(dataset_name,delta, seed):
    print(f'we are running dataset_name: {dataset_name} with seed: {seed}')
    device = torch.device('cpu')

    # Set a seed for random number generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



    # Import dataset
    dataset = FairnessDataset(dataset=dataset_name, seed = seed, device=device)
    dataset.normalize()
    input_dim = dataset.XZ_train.shape[1]
    n_epochs, lr, batch_size= get_training_parameters(dataset_name)


    # Create a classifier model

    net = Classifier(n_inputs=input_dim)
    net = net.to(device)

    # Set an optimizer and decay rate
    lr_decay = 0.98
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # Fair classifier training
    Result = FUDS_multiclass(dataset=dataset,dataset_name=dataset_name,
                     net=net,
                     optimizer=optimizer,lr_schedule=lr_schedule,delta = delta,
                     device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)

    print(Result)
    Result.to_csv(f'Result/FUDS/result_of_{dataset_name}_with_seed_{seed}_para_{int(delta*1000)}')











