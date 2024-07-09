import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from feeders import DataInterface
from models import ModelInterface
from utils.utils import *


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='test', type=str)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--log_path', type=str,
                        default=r'logs/Camelyon17')
    parser.add_argument('--config', default='config/Camelyon17.yaml', type=str)
    parser.add_argument('--gpus', default=[0])
    parser.add_argument('--fold', default=0)
    args = parser.parse_args()
    return args

def main(cfg):
    print('-------------------------------Initialize')
    pl.seed_everything(cfg.General.seed)
    cfg.load_loggers = load_loggers(cfg)
    cfg.callbacks = load_callbacks(cfg)

    print('-------------------------------Define Data')
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                          'train_num_workers': cfg.Data.train_dataloader.num_workers,
                          'test_batch_size': cfg.Data.test_dataloader.batch_size,
                          'test_num_workers': cfg.Data.test_dataloader.num_workers,
                          'dataset_name': cfg.Data.dataset_name,
                          'dataset_cfg': cfg.Data, }
    dm = DataInterface(**DataInterface_dict)

    print('-------------------------------Define Model')
    cfg.Loss.n_classes  = cfg.Model.n_classes
    ModelInterface_dict = {'model': cfg.Model,
                           'loss': cfg.Loss,
                           'optimizer': cfg.Optimizer,
                           'data': cfg.Data,
                           'log': cfg.log_path
                           }
    model = ModelInterface(**ModelInterface_dict)
    torch.use_deterministic_algorithms(False)

    print('-------------------------------Instantiate Trainer')
    trainer = Trainer(
        num_sanity_val_steps=0,
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs=cfg.General.epochs,
        accelerator='gpu', devices='1',
        precision=cfg.General.precision,
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )

    if cfg.General.server == 'train':
        if cfg.resume:
            model_paths = list(list(cfg.log_path.glob('*.ckpt')))
            model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
            path = model_paths[0]
            print(path)
            model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
        trainer.fit(model=model, datamodule=dm)
        cfg.resume = False
    else:
        torch.use_deterministic_algorithms(False)  # 添加部分
        model_paths = list(list(cfg.log_path.glob('*.ckpt')))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':
    args = make_parse()
    cfg = read_yaml(args.config)
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.resume = args.resume
    cfg.test_path = args.test_path

    for i in range(0, cfg.Data.nfold):
        print('-------------------------------fold{}'.format(i))
        cfg.Data.fold = i
        main(cfg)
