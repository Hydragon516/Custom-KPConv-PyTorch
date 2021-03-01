import time
import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
import numpy as np

from datasets.Custom_Dataset import *
from torch.utils.data import DataLoader
from models.architectures import KPCNN

import config.config as config

# import warnings
# warnings.filterwarnings("ignore")

def train(epoch, trainloader, optimizer, model, device):
    model.train()

    avg_loss = 0
    avg_acc = 0

    for idx, batch in enumerate(trainloader):
        batch.to(device)
        optimizer.zero_grad() 
        
        outputs = model(batch)

        loss = model.loss(outputs, batch.labels)
        acc = model.accuracy(outputs, batch.labels)

        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), config.TRAIN['grad_clip_norm'])

        avg_loss += loss.item()
        avg_acc += acc

        optimizer.step()
        torch.cuda.synchronize(device)

        if (idx % config.TRAIN['print_freq'] == 0) and (idx > 0):
            avg_loss = avg_loss / config.TRAIN['print_freq']
            avg_acc = avg_acc / config.TRAIN['print_freq']

            print(
                "Epoch: #{0} Batch: {1}\t"
                "Lr: {lr:.6f}\t"
                "LOSS: {loss:.4f}\t"
                "ACC: {acc:.4f}\t"
                .format(epoch, idx, lr=optimizer.param_groups[-1]['lr'], \
                    loss=avg_loss, acc=avg_acc)
            )

            avg_loss = 0
            avg_acc = 0


def valid(testloader, model, device):
    model.eval()

    avg_acc = 0

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            batch.to(device)
            
            outputs = model(batch)
            acc = model.accuracy(outputs, batch.labels)

            avg_acc += acc

        avg_acc = avg_acc / (idx + 1)

        print("TEST ACC :", str(avg_acc))
        
        return avg_acc


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("Load dataset...")
    training_dataset = CustomDataset(train=True, subsampling=True, first_subsampling_dl=config.SETTING['first_subsampling_dl'])
    test_dataset = CustomDataset(train=False, subsampling=True, first_subsampling_dl=config.SETTING['first_subsampling_dl'])

    training_sampler = CustomDataSampler(training_dataset, balance_labels=True)
    test_sampler = CustomDataSampler(test_dataset, balance_labels=True)

    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=ModelNet40Collate,
                                 num_workers=config.SETTING['input_threads'],
                                 pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=ModelNet40Collate,
                             num_workers=config.SETTING['input_threads'],
                             pin_memory=True)
    
    
    print("ok!")

    print("Check device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("ok!")

    print("Load model...")
    model = KPCNN()
    model = model.to(device)
    print("ok!")

    print("Load optimizer...")
    deform_params = [v for k, v in model.named_parameters() if 'offset' in k]
    other_params = [v for k, v in model.named_parameters() if 'offset' not in k]
    deform_lr = config.TRAIN['learning_rate'] * config.TRAIN['deform_lr_factor']
    optimizer = torch.optim.SGD([{'params': other_params},
                                {'params': deform_params, 'lr': deform_lr}],
                                lr=config.TRAIN['learning_rate'],
                                momentum=config.TRAIN['momentum'],
                                weight_decay=config.TRAIN['weight_decay'])
    print("ok!")

    # Starting training 
    print("Starting training... ")

    best = 1
    
    for epoch in range(config.TRAIN['epoch']):
        train(epoch, training_loader, optimizer, model, device)
        acc = valid(test_loader, model, device)

        # if rmse < best:
        #     torch.save({
        #         "epoch": epoch, 
        #         "model_state_dict": model.state_dict(),
        #         "optim_state_dict": optimizer.state_dict()
        #     }, "ckpt_{}.pth".format(epoch))

        #     best = rmse


if __name__ == "__main__":
    main()