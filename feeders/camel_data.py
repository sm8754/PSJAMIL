import random
import os
import torch
torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
from pathlib import Path

import torch.utils.data as data
from torch.utils.data import dataloader


class CamelData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):

        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        self.reverse_prob = self.dataset_cfg.type
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        self.shuffle = self.dataset_cfg.data_shuffle
        self.data = []

        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()

        if state == 'val':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()

        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        features = torch.load(full_path)

        if self.shuffle == True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]
        return features, label

