import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader.dataloader import CustomDataset
from utils import cal_disparity,fint_thresholds,cal_acc

from DataLoader.dataloader import FairnessDataset
import torch.optim as optim
from models import Classifier
def get_level(dataset_name):
    if dataset_name == 'AdultCensus':
        level_list = np.arange(10)/50
    if dataset_name == 'COMPAS':
        level_list  = 0.3 * np.arange(10) / 10
    if dataset_name == 'Lawschool':
        level_list = np.arange(10) / 125

    return level_list

def FPIR(dataset,dataset_name,net, optimizer, lr_schedule,fairness, device, n_epochs=200, batch_size=2048, seed=0):

    # Retrieve train/test splitted pytorch tensors
    training_tensors, testing_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = training_tensors
    X_test, Y_test, Z_test, XZ_test = testing_tensors


###Minibatch for SGD########
    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    if batch_size == 'full':
        batch_size_ = XZ_train.shape[0]
    elif isinstance(batch_size, int):
        batch_size_ = batch_size
    data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)



#### Network Training##########
    loss_function = nn.BCELoss()
    with tqdm(range(n_epochs)) as epochs:
        epochs.set_description(f"Training the classifier with dataset: {dataset_name}, seed: {seed}")
        net.train()

        for epoch in epochs:
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


##########performance of fair Classifier on test sets###########

    eta_train = net(XZ_train).detach().cpu().numpy().squeeze()

    eta_test = net(XZ_test).detach().cpu().numpy().squeeze()

    Y_train_np = Y_train.clone().detach().numpy()
    Z_train_np = Z_train.clone().detach().numpy()
    Y_test_np = Y_test.clone().detach().numpy()
    Z_test_np = Z_test.clone().detach().numpy()
    df_test = pd.DataFrame()


    level_list = get_level(dataset_name)
    for level in level_list:
        t1, t0 = fint_thresholds(eta_train,Y_train_np,Z_train_np,level)
        acc = cal_acc(eta_test,Y_test_np,Z_test_np,t1,t0)
        disparity = cal_disparity(eta_test,Z_test_np,t1,t0)

        data = [seed,dataset_name,level,acc, np.abs(disparity)]
        columns = ['seed','dataset','level','acc', 'disparity']


        temp_dp = pd.DataFrame([data], columns=columns)
        df_test= pd.concat([df_test,temp_dp])
    return df_test




def get_training_parameters(dataset_name):
    if dataset_name == 'AdultCensus':
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






def training_FPIR(dataset_name,seed):
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

    Result = FPIR(dataset=dataset,dataset_name=dataset_name,
                     net=net,optimizer=optimizer,
                     lr_schedule=lr_schedule,fairness = 'DP',
                     device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print(Result)

    Result.to_csv(f'Result/FPIR/result_of_{dataset_name}_with_seed_{seed}')










