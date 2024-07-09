import torch
import torch.nn as nn
import argparse

from PSJALoss import PSJALoss

def create_loss(args):
    conf_loss = args.base_loss
    if hasattr(nn, conf_loss): 
        loss = getattr(nn, conf_loss)()
    elif conf_loss == "PSJALoss":
        loss = PSJALoss(args)
    else:
        assert False and "Invalid loss"
        raise ValueError
    return loss

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-loss', default='CrossEntropyLoss',type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = make_parse()
    myloss = create_loss(args)
    data = torch.randn(2, 3)
    label = torch.empty(2, dtype=torch.long).random_(3)
    loss = myloss(data, label)