# Patch-Slide Discriminative Joint Learning for Weakly-Supervised Whole Slide Image Representation and Classification [MICCAI 2024]

**Abstract:** In computational pathology, Multiple Instance Learning (MIL) is widely ap-plied for classifying Giga-pixel whole slide images (WSIs) with only image-level labels. Due to the size and prominence of positive areas varying signif-icantly across different WSIs, it is difficult for existing methods to learn task-specific features accurately. Additionally, subjective label noise usually affects deep learning frameworks, further hindering the mining of discrimi-native features. To address this problem, we propose an effective theory that optimizes patch and WSI feature extraction jointly, enhancing feature dis-criminability. Powered by this theory, we develop an angle-guided MIL framework called PSJA-MIL, effectively leveraging features at both levels. We also focus on eliminating noise between instances and emphasizing fea-ture enhancement within WSIs. We evaluate our approach on Camelyon17 and TCGA-Liver datasets, comparing it against state-of-the-art methods. The experimental results show significant improvements in accuracy and general-izability, surpassing the latest methods by more than 2%.


![overview](picture/ImplementationProcess.jfif)

## Dataset
[Camelyon17](https://camelyon17.grand-challenge.org/Data/). This dataset identifies lymph node metastases containing normal and tumor cate-gories. Among them, tumor samples can be divided into three types. Isolated tu-mor cells (ITC) is the minorest type of metastasis, smaller than 0.2 mm or less than 200 cells, which is very challenging. Since only the annotations for the training set are publicly accessible, we used the training set for experiments. These data encompass 500 WSIs from 100 patients in 5 medical centers. 

[TCGA-Liver](https://www.cancer.gov/tcga). This dataset is collected from The Cancer Genome Atlas (TCGA) Data Portal, containing two categories: Liver Cancer (LIHC) and Bile Duct Can-cer (CHOL). There is a severe imbalance in the dataset, including 379 LIHC WSIs and 36 CHOL WSIs. This poses a significant challenge to the feature learning. Each dataset is randomly split into a training-validation set and a test set in a 7:3 ratio. The training validation set is performed five-fold cross-validation, and the test set is used to report and compare model performance.

## Data Preprocess
1.See [CLAM](https://github.com/mahmoodlab/CLAM) for WSI processing. Specific settings are as follows:
- All tumor regions in the WSIs are segmented into 256×256 patches. 
- For Camelyon17, the magnification is 40x, and for TCGA-LIBD, it is 20x.
- Extract 1024-dim features from the patches: 
  Mode 1: Using a ResNet50 pretrained on ImageNet;
  Mode 2: Using a [KimiaNet](https://github.com/KimiaLabMayo/KimiaNet) pretrained on TCGA slides.
- Keep only .pt files.(.pt file for each slide, containing the patch features)

2.Summarize the sample names and categories in the dataset into a .csv file with columns for 'sample' and 'label'.

3.[folddata.py](datasets/folddata.py) are used to divide the data set into training sets, validation sets, and test sets. Each dataset is randomly split into a training-validation set (The default is five-fold cross-validation) and a test set in a 7:3 ratio. 

4.Generate the following folder structure at the specified datasets:

```bash
datasets/
    ├──Camelyon17/
        ├── dataset_csv
                ├── fold0.csv
                ├── fold1.csv
                └── ...
        ├── pt_files
                ├── slide_1.pt
                ├── slide_2.pt
                └── ...
        └── class.csv
```

## Installation

- GPU(a single RTX 4090 GPU)
- Cuda(11.7)
- Python (3.9), PyTorch (1.13.0), pytorch-lightning (2.0.9)

## Train
The Lookahead-RAdam optimizer is used with an initial learn-ing rate of 0.0001 and a batch size 1. 
```python
python train.py --stage='train' --config='config/Camelyon17.yaml'  --gpus=0 --fold=0
```

## Test
Accuracy (ACC) and area under the ROC curve (AUC) are used as evaluation metrics.
```python
python train.py --stage='test' --config='config/Camelyon17.yaml'  --gpus=0 --fold=0
```

