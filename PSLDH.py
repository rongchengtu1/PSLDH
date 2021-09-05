import pickle
import os
import argparse
import logging
import torch
import time
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.data_processing as dp
import utils.hash_model as image_hash_model
import utils.label_hash_model as label_hash_model
import utils.calc_hr as calc_hr
import torch.nn as nn


def load_label(label_filename, ind, DATA_DIR):
    label_filepath = os.path.join(DATA_DIR, label_filename)
    label = np.loadtxt(label_filepath, dtype=np.int64)
    ind_filepath = os.path.join(DATA_DIR, ind)
    fp = open(ind_filepath, 'r')
    ind_list = [x.strip() for x in fp]
    fp.close()
    ind_np = np.asarray(ind_list, dtype=np.int)
    ind_np = ind_np - 1
    ind_label = label[ind_np, :]
    return torch.from_numpy(ind_label)


def GenerateCode(model_hash, data_loader, num_data, bit, k=0):
    B = np.zeros((num_data, bit), dtype=np.float32)
    kk = 1
    for iter, data in enumerate(data_loader, 0):
        data_img, _, _, _, data_ind = data
        data_img = Variable(data_img.cuda())
        if k == 0:
            out = model_hash(data_img)
            if kk:
                # print(out.item())
                kk = 0
            B[data_ind.numpy(), :] = torch.sign(out.data.cpu()).numpy()
    return B

def PSLDH_Algo(code_length):
    DATA_DIR = '/data/home/trc/mat/imagenet'
    LABEL_FILE = 'label_hot.txt'
    IMAGE_FILE = 'images_name.txt'
    DATABASE_FILE = 'database_ind_ph.txt'
    TRAIN_FILE = 'train_ind_ph.txt'
    TEST_FILE = 'test_ind_ph.txt'
    data_set = 'imagenet_vgg11'
    data_name = './label_codes/imagenet'
    top_k = 1000

    os.environ['CUDA_VISIBLE_DEVICES'] = str(5)

    batch_size = 80
    epochs = 150
    learning_rate = 0.001 #0.05
    weight_decay = 10 ** -5
    model_name = 'vgg11'

    alpha = 0.05
    beta = 0.01
    lamda = 0.01 #50
    gamma = 0.2
    sigma = 0.2
    print("*"*10, learning_rate, alpha, beta, lamda, sigma, gamma, code_length, top_k, data_set, "*"*10)
    ### data processing



    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dset_database = dp.DatasetProcessing(
        DATA_DIR, IMAGE_FILE, LABEL_FILE, DATABASE_FILE, transformations)

    dset_train = dp.DatasetProcessing(
        DATA_DIR, IMAGE_FILE, LABEL_FILE, TRAIN_FILE, transformations, 0)

    dset_test = dp.DatasetProcessing(
        DATA_DIR, IMAGE_FILE, LABEL_FILE, TEST_FILE, transformations)

    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)
    database_loader = DataLoader(dset_database,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4
                             )

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )
    test_loader = DataLoader(dset_test,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )
    train_labels = load_label(LABEL_FILE, TRAIN_FILE, DATA_DIR)
    database_labels = load_label(LABEL_FILE, DATABASE_FILE, DATA_DIR)
    test_labels = load_label(LABEL_FILE, TEST_FILE, DATA_DIR)
    label_size = test_labels.size()
    nclass = label_size[1]
    if os.path.exists(data_name + '_label_code_' + str(code_length) + '.pkl'):
        with open(data_name + '_label_code_' + str(code_length) + '.pkl', 'rb') as f:
            label_code = pickle.load(f)
    else:
        label_model = label_hash_model.Label_net(nclass, code_length)
        label_model.cuda()

        optimizer_label = optim.SGD(label_model.parameters(), lr=0.001, weight_decay=weight_decay)
        scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_label, step_size=100, gamma=0.1, last_epoch=-1)

        labels = torch.zeros((nclass, nclass)).type(torch.FloatTensor).cuda()
        for i in range(nclass):
            labels[i, i] = 1

        labels = Variable(labels)
        one_hot = Variable(torch.ones((1, nclass)).type(torch.FloatTensor).cuda())
        I = Variable(torch.eye(nclass).type(torch.FloatTensor).cuda())
        relu = nn.ReLU()
        for i in range(200):
            scheduler_l.step()
            code = label_model(labels)
            loss1 = relu((code.mm(code.t()) - code_length * I))
            loss1 = loss1.pow(2).sum() / (nclass * nclass)
            loss_b = one_hot.mm(code).pow(2).sum() / nclass
            re = (torch.sign(code) - code).pow(2).sum() / nclass
            loss = loss1 + alpha * loss_b + beta * re
            optimizer_label.zero_grad()
            loss.backward()
            optimizer_label.step()
        label_model.eval()
        code = label_model(labels)
        label_code = torch.sign(code)
        with open(data_name + '_label_code_' + str(code_length) + '.pkl', 'wb') as f:
            pickle.dump(label_code, f)

    hash_model = image_hash_model.HASH_Net(model_name, code_length)
    hash_model.cuda()
    optimizer_hash = optim.SGD(hash_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_hash, step_size=100, gamma=0.3, last_epoch=-1)
    for epoch in range(epochs):
        scheduler.step()
        epoch_loss = 0.0
        epoch_loss_r = 0.0
        epoch_loss_e = 0.0
        ## training epoch
        for iter, traindata in enumerate(train_loader, 0):
            train_img, train_label, batch_ind = traindata
            train_label = torch.squeeze(train_label)
            train_img = Variable(train_img.cuda())
            train_label = Variable(train_label.type(torch.FloatTensor).cuda())
            the_batch = len(batch_ind)
            hash_out = hash_model(train_img)
            logit = hash_out.mm(label_code.t())

            our_logit = torch.exp((logit - sigma * code_length) * gamma) * train_label
            mu_logit = torch.exp(logit * (1 - train_label) * gamma).sum(1).view(-1, 1).expand(the_batch, train_label.size()[1]) + our_logit
            loss = - ((torch.log(our_logit / mu_logit + 1 - train_label)).sum(1) / train_label.sum(1)).sum()

            Bbatch = torch.sign(hash_out)
            regterm = (Bbatch - hash_out).pow(2).sum()
            loss_all = loss / the_batch + regterm * lamda / the_batch

            optimizer_hash.zero_grad()
            loss_all.backward()
            optimizer_hash.step()
            epoch_loss += loss_all.item()
            epoch_loss_e += loss.item() / the_batch
            epoch_loss_r += regterm.item() / the_batch
        print('[Train Phase][Epoch: %3d/%3d][Loss_i: %3.5f, Loss_e: %3.5f, Loss_r: %3.5f]' %
              (epoch + 1, epochs, epoch_loss / len(train_loader), epoch_loss_e / len(train_loader),
               epoch_loss_r / len(train_loader)))

        if (epoch + 1) % 50 == 0:
            hash_model.eval()
            qi = GenerateCode(hash_model, test_loader, num_test, bit)
            ri = GenerateCode(hash_model, database_loader, num_database, bit)
            map = calc_hr.calc_topMap(qi, ri, test_labels.numpy(), database_labels.numpy(), top_k)
            print('test_map:', map)
    '''
    training procedure finishes, evaluation
    '''


if __name__=="__main__":
    bits = [64, 48, 32, 16]
    for bit in bits:
        PSLDH_Algo(bit)
