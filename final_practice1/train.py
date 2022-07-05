import json
from collections import namedtuple
from importlib import import_module
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np; np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    args = parser.parse_args()
    return args

def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def train(num_epochs, model, train_loader, val_loader, optimizer, criterion, val_term, scheduler=None):
    print('Start training...')
    start_epoch = 0
    best_accuracy = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        # 학습 코드를 작성하시오.
        
        pbar = tqdm(enumerate(train_loader), total = len(train_loader))
        for step, input in pbar:


            description =  f'Epoch [{epoch}/{num_epochs}], Step [{step+1}/{len(train_loader)}]: ' 
            description += f'running Loss: {round(running_loss,4)}'
            pbar.set_description(description)

        if epoch % val_term == 0:
            # 검증 코드를 작성하시오. # 수빈
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    ouput = model(x)
                    val_loss += criterion(ouput, y).item()
                    prediction = output.max(1, keepdim = True)[1]
                    correct += prediction.eq(y.view_as(prediction)).sum().item()
                    
            accuracy_ = 100. * correct / len(val_loader.dataset)
            print(f'Epoch [{epoch} / {num_epochs}], Accuracy: {accuracy_}')
            if accuracy_ > best_accuracy:
                print(f'Best Performance at epoch: {epoch}')
    
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(accuracy_)
                else:
                    scheduler.step()

def main():
    args = arg_parse()
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))
    
    fix_seed(cfgs.seed)

    # dataset & data loader
    train_dataloader_module = getattr(import_module("datasets"), cfgs.train_dataloader.name)
    train_dataloader = train_dataloader_module(**cfgs.train_dataloader.args._asdict())
    
    valid_dataloader = train_dataloader.split_validation()

    # model
    model_module = getattr(import_module("model"), cfgs.model.name)
    model = model_module(**cfgs.model.args._asdict()).to(device)

    # criterion
    criterion_module = getattr(import_module("torch.nn"), cfgs.criterion.name)
    criterion = criterion_module(**cfgs.criterion.args._asdict())

    # optimizer
    optimizer_module = getattr(import_module("torch.optim"), cfgs.optimizer.name)
    optimizer = optimizer_module(model.parameters(), **cfgs.optimizer.args._asdict())

    # scheduler
    try:
        scheduler_module = getattr(import_module("torch.optim.lr_scheduler"), cfgs.scheduler.name)
        scheduler = scheduler_module(optimizer, **cfgs.scheduler.args._asdict())
    except AttributeError :
            print('There is no Scheduler!')
            scheduler = None

    train_args = {
        'num_epochs': cfgs.num_epochs, 
        'model': model, 
        'train_loader': train_dataloader, 
        'val_loader': valid_dataloader, 
        'optimizer': optimizer, 
        'criterion': criterion, 
        'val_term': cfgs.val_term, 
        'scheduler': scheduler
    }

    train(**train_args)

if __name__ == '__main__:
    main()
