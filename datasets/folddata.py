import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from random import sample

data_dir = './Camelyon17/pt_files'
csv_path = './Camelyon17/dataset_csv'
information = pd.read_csv(r'./Camelyon17/class.csv')

samples=[]
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.pt'):
            samples.append(file)

information = pd.DataFrame(information)

test_split = 0.3
total_data_n = len(samples)
test_n = int(total_data_n * test_split)

test = sample(samples, test_n)
train_val = list(set(samples)-set(test))

data = [[],[],[],[]]
datasets=[train_val,test]

for s, set in enumerate(datasets):
    for f,file in enumerate(set):
        file = file[:-3]
        index = information[information.files == file].index
        classname = information.loc[index, 'label'].values
        data[s * 2].append(file)
        data[s * 2 + 1].append(int(classname))

skf = StratifiedKFold(n_splits=5)
valset = []
valtag = []
trainset = []
traintag = []

for i, (train_idx, val_idx) in enumerate(skf.split(data[0], data[1])):
    for idx in val_idx:
        valset.append(data[0][idx])
        valtag.append(data[1][idx])
    for idx in train_idx:
        trainset.append(data[0][idx])
        traintag.append(data[1][idx])

    df = pd.concat(
        [pd.DataFrame({'train': trainset}), pd.DataFrame({'train_label': traintag}),
         pd.DataFrame({'val': valset}), pd.DataFrame({'val_label':valtag}),
         pd.DataFrame({'test': data[2]}), pd.DataFrame({'test_label': data[3]})],
        axis=1)
    df.fillna(0)
    df.to_csv(os.path.join(csv_path, 'fold{}.csv'.format(str(i))), index=True)

