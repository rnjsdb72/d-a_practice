import os
import re
import sys
import glob
import json
import shutil
from pathlib import Path
from collections import namedtuple
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from utils import * 

from tqdm import tqdm
from datetime import datetime
import numpy as np; np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def train(num_epochs, model, train_loader, val_loader, optimizer, criterion, val_term, scheduler=None):
    print('Start training...')
    start_epoch = 0
    best_accuracy = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        # 학습 코드를 작성하시오.

        if epoch % val_term == 0:
            # 검증 코드를 작성하시오.

            accuracy_ = 
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

    # 데이터 불러오기

    # 모델 불러오기

    # optimizer 불러오기

    # criterion 불러오기

    # scheduler 불러오기

    train_args = {

    }

    train(**train_args)

if __name__ == '__main__()':
    main()