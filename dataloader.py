import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms

Labels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK", "VA", "ND"]

class MultiLabelDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, onlyDefects=False):
        super(MultiLabelDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()
        self.class_weights = self.getClassWeights()

    def loadAnnotations(self):
        #gtPath = os.path.join(self.annRoot, "{}13.csv".format(self.split)) # use this will cause error, don't know why.
        gtPath = os.path.join(self.annRoot, "Train13.csv")
        # gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = self.LabelNames + ["Filename", "Defect"])
        #print("the gtPath is {}".format(gtPath))
        if self.onlyDefects:
            gt = gt[gt["Defect"] == 1]

        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.LabelNames].values
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        #print("the self.imgPath is {}".format(self.imgPaths))
        #print("the img is {} and the path is {}".format(self.annRoot, path))
        img = self.loader(os.path.join(self.imgRoot, path))
        
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index, :]

        return img, target, path

    def getClassWeights(self):
        data_len = self.labels.shape[0]
        class_weights = []

        for defect in range(self.num_classes):
            pos_count = len(self.labels[self.labels[:,defect] == 1])
            neg_count = data_len - pos_count

            class_weight = neg_count/pos_count if pos_count > 0 else 0
            class_weights.append(np.asarray([class_weight]))
        return torch.as_tensor(class_weights).squeeze()


class MultiLabelDatasetInference(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, onlyDefects=False):
        super(MultiLabelDatasetInference, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "{}13.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename"])
        #print("the gtpath is {}".format(gtPath))
        self.imgPaths = gt["Filename"].values
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        return img, path


class BinaryRelevanceDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, defect=None):
        super(BinaryRelevanceDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.defect = defect

        assert self.defect in self.LabelNames

        self.num_classes = 1

        self.loadAnnotations()
        self.class_weights = self.getClassWeights()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "{}13.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename", self.defect])

        self.imgPaths = gt["Filename"].values
        self.labels =  gt[self.defect].values.reshape(self.imgPaths.shape[0], 1)
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]

        return img, target, path

    def getClassWeights(self):
        pos_count = len(self.labels[self.labels == 1])
        neg_count = self.labels.shape[0] - pos_count
        class_weight = np.asarray([neg_count/pos_count])

        return torch.as_tensor(class_weight)


class BinaryDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader):
        super(BinaryDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.num_classes = 1

        self.loadAnnotations()
        self.class_weights = self.getClassWeights()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "{}13.csv".format(self.split))
      # gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename", "Defect"])

        self.imgPaths = gt["Filename"].values
        self.labels =  gt["Defect"].values.reshape(self.imgPaths.shape[0], 1)
        print(self.labels.shape)
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]

        return img, target, path

    def getClassWeights(self):
        pos_count = len(self.labels[self.labels == 1])
        neg_count = self.labels.shape[0] - pos_count
        class_weight = np.asarray([neg_count/pos_count])

        return torch.as_tensor(class_weight)



if __name__ == "__main__":
    
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor()]
    )

    
    train = MultiLabelDataset(annRoot="D:\\Documents\\VS2022Projects\\sewer-ml\\annotations", imgRoot="D:\\Documents\\VS2022Projects\\sewer-ml\\Data", split="Train", transform=transform)
    train_defect = MultiLabelDataset(annRoot="D:\\Documents\\VS2022Projects\\sewer-ml\\annotations", imgRoot="D:\\Documents\\VS2022Projects\\sewer-ml\\Data", split="Train", transform=transform, onlyDefects=True)
    binary_train = BinaryDataset(annRoot="D:\\Documents\\VS2022Projects\\sewer-ml\\annotations", imgRoot="D:\\Documents\\VS2022Projects\\sewer-ml\\Data", split="Train", transform=transform)
    binary_relevance_train = BinaryRelevanceDataset(annRoot="D:\\Documents\\VS2022Projects\\sewer-ml\\annotations", imgRoot="D:\\Documents\\VS2022Projects\\sewer-ml\\Data", split="Train", transform=transform, defect="RB")

    print(len(train), len(train_defect), len(binary_train), len(binary_relevance_train))
    print(train.class_weights, train_defect.class_weights, binary_train.class_weights, binary_relevance_train.class_weights)

    