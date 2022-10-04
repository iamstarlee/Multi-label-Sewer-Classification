
from torch.utils.data import DataLoader
from dataloader import MultiLabelDataset, BinaryDataset, BinaryRelevanceDataset
import pytorch_lightning as pl


class MultiLabelDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, workers=4, ann_root="D:\\Documents\\VS2022Projects\\sewer-ml\\annotations", data_root="D:\\Documents\\VS2022Projects\\sewer-ml\\Data\\train13", only_defects=False, train_transform = None, eval_transform = None):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root
        self.only_defects = only_defects

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_dataset = MultiLabelDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform, onlyDefects=self.only_defects)
            self.val_dataset = MultiLabelDataset(self.ann_root, self.data_root, split="Val", transform=self.eval_transform, onlyDefects=self.only_defects)
        if stage == 'test':
            self.test_dataset = MultiLabelDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform, onlyDefects=self.only_defects)

        self.num_classes = self.train_dataset.num_classes
        self.class_weights = self.train_dataset.class_weights
        self.LabelNames = self.train_dataset.LabelNames

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True, drop_last = True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl


class BinaryDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, workers=4, ann_root="D:\\Documents\\VS2022Projects\\sewer-ml\\annotations", data_root="D:\\Documents\\VS2022Projects\\sewer-ml\\Data\\train13", train_transform = None, eval_transform = None):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_dataset = BinaryDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform)
            self.val_dataset = BinaryDataset(self.ann_root, self.data_root, split="Val", transform=self.eval_transform)
        if stage == 'test':
            self.test_dataset = BinaryDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform)

        self.num_classes = self.train_dataset.num_classes
        self.class_weights = self.train_dataset.class_weights
        self.LabelNames = self.train_dataset.labels # self.train_dataset.LabelNames

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl



class BinaryRelevanceDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, workers=4, ann_root="D:\\Documents\\VS2022Projects\\sewer-ml\\annotations", data_root="D:\\Documents\\VS2022Projects\\sewer-ml\\Data\\train13", defect=None, train_transform = None, eval_transform = None):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.ann_root = ann_root
        self.data_root = data_root
        self.defect = defect

        self.train_transform = train_transform
        self.eval_transform = eval_transform

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        # split dataset
        if stage == 'fit':
            self.train_dataset = BinaryRelevanceDataset(self.ann_root, self.data_root, split="Train", transform=self.train_transform, defect=self.defect)
            self.val_dataset = BinaryRelevanceDataset(self.ann_root, self.data_root, split="Val", transform=self.eval_transform, defect=self.defect)
        if stage == 'test':
            self.test_dataset = BinaryRelevanceDataset(self.ann_root, self.data_root, split="Test", transform=self.eval_transform, defect=self.defect)

        self.num_classes = self.train_dataset.num_classes
        self.class_weights = self.train_dataset.class_weights
        self.LabelNames = self.train_dataset.LabelNames

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.workers, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.workers, pin_memory=True)
        return test_dl
