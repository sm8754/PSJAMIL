from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import pandas as pd
import inspect
import importlib
import random
import numpy as np

import os
import torch
torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torchmetrics
import pytorch_lightning as pl

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

from optimizers import create_optimizer
from losses import create_loss

class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer,**kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']
        self.validation_step_outputs = []
        self.test_step_outputs = []
        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(task = "multiclass", num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task = "multiclass",
                                                                            num_classes = self.n_classes,
                                                                           average='micro'),
                                                     torchmetrics.CohenKappa(task = "multiclass", num_classes = self.n_classes),
                                                     torchmetrics.F1Score(task = "multiclass", num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(task = "multiclass", average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(task = "multiclass", average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(task = "multiclass", average = 'macro',
                                                                            num_classes = self.n_classes)])
        else : 
            self.AUROC = torchmetrics.AUROC(task='binary',num_classes=2, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary',num_classes = 2,
                                                                           average = 'micro'),
                                                     torchmetrics.CohenKappa(task='binary',num_classes = 2),
                                                     torchmetrics.F1Score(task='binary',num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(task='binary',average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(task='binary',average = 'macro',
                                                                            num_classes = 2)])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0

    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label,train=True)
        logits_ori = results_dict['logits_ori']
        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']
        score = results_dict['atten_score']
        fea = results_dict['fea']

        l2_wei = 0.0
        for param in self.model.parameters():
            l2_wei += torch.norm(param) ** 2

        loss = self.loss(fea, logits[:,int(label)],logits_ori, score, label,l2_wei*0.5)

        Y_hat = int(Y_hat)
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        return {'loss': loss}

    def on_train_epoch_end(self):
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]


    def validation_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label,train=False)#, fea,weight,score
        logits_ori = results_dict['logits_ori']
        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']
        Y_prob = results_dict['Y_prob']
        score = results_dict['atten_score']
        fea = results_dict['fea']
        del results_dict['logits_ori']
        del results_dict['logits']
        del results_dict['Y_prob']
        del results_dict['atten_score']
        del results_dict['fea']
        results_dict['label'] = label


        l2_wei = 0.0
        for param in self.model.parameters():
            l2_wei += torch.norm(param) ** 2

        loss = self.loss(fea, logits[:, int(label)], logits_ori, score, label, l2_wei * 0.5)
        results_dict['loss'] = loss
        self.validation_step_outputs.append(results_dict)

        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def on_validation_epoch_end(self):
        loss = torch.stack([x['loss'] for x in self.validation_step_outputs], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in self.validation_step_outputs])
        target = torch.stack([x['label'] for x in self.validation_step_outputs], dim=0)

        self.validation_step_outputs.clear()
        torch.use_deterministic_algorithms(False)  # 添加部分

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze(), target.squeeze()),
                      on_epoch=True, logger=True)

        # ---->acc log
        early_stop = [0,0,0]
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
                if int(acc) == 1:
                    early_stop[c]=1
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        # ---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count + 1
            random.seed(self.count * 50)


    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        data, label = batch
        results_dict, fea,weight,score = self.model(data=data, label=label,train=False)#

        Y_hat = results_dict['Y_hat']
        results_dict['label'] = label
        del results_dict['logits_ori']
        del results_dict['logits']
        del results_dict['atten_score']
        del results_dict['fea']

        self.test_step_outputs.append(results_dict)
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        return {'Y_hat' : Y_hat, 'label' : label}

    def on_test_epoch_end(self):
        probs = torch.cat([x['Y_prob'] for x in self.test_step_outputs], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in self.test_step_outputs])
        target = torch.stack([x['label'] for x in self.test_step_outputs], dim=0)
        self.test_step_outputs.clear()

        y_true_binary = label_binarize(target.squeeze().cpu(), classes=[0, 1, 2])  # 计算宏平均AUC
        auc_scores = []
        probs = probs.cpu().numpy()
        for i in range(3):
            auc_score = roc_auc_score(y_true_binary[:, i], probs[:, i])
            auc_scores.append(auc_score)
        auc = np.mean(auc_scores)
        metrics = self.test_metrics(max_probs.squeeze(), target.squeeze())
        metrics['auc'] = auc

        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            if not isinstance(values, np.float64):
                metrics[keys] = values.cpu().numpy()

        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        result = pd.DataFrame([metrics])
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        result.to_csv(self.log_path / 'result-{}.csv'.format(self.radius))

    def load_model(self):
        name = self.hparams.model.name
        print(name)
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)