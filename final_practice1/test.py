from train import arg_parse
from tqdm import tqdm
from importlib import import_module
from collections import namedtuple
import json

import torch

def test():
    args = arg_parse()
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    # data_loader
    test_dataloader_module = getattr(import_module("datasets"), cfgs.test_dataloader.name)
    test_dataloader = test_dataloader_module(**cfgs.test_dataloader.args._asdict())

    # model
    model_module = getattr(import_module('model'), cfgs.model.name)
    model = model_module(**cfgs.model.args.as_dict()).to(device)

    # criterion
    criterion_module = getattr(import_module("torch.nn"), cfgs.criterion.name)
    criterion = criterion_module(**cfgs.criterion.args._asdict())

    # test 코드를 작성하시오.
    


    return loss, accuracy

if __name__ == '__main__':
    loss, accuracy = test()
    print(f'Test Loss: {round(loss, 4)},\t Test Accuracy: {round(accuracy, 4)}')