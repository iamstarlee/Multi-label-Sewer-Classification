import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn

from torchvision import models as torch_models
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_datamodules import MultiLabelDataModule, BinaryDataModule, BinaryRelevanceDataModule
import sewer_models
import ml_models



class MultiLabelModel(pl.LightningModule):

    TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__ if name.islower() and not name.startswith("__") and callable(torch_models.__dict__[name]))
    SEWER_MODEL_NAMES = sorted(name for name in sewer_models.__dict__ if name.islower() and not name.startswith("__") and callable(sewer_models.__dict__[name]))
    MULTILABEL_MODEL_NAMES = sorted(name for name in ml_models.__dict__ if name.islower() and not name.startswith("__") and callable(ml_models.__dict__[name]))
    MODEL_NAMES =  TORCHVISION_MODEL_NAMES + SEWER_MODEL_NAMES + MULTILABEL_MODEL_NAMES
    

    def __init__(self, model = "resnet18", num_classes=18, learning_rate=1e-2, momentum=0.9, weight_decay = 0.0001, criterion=torch.nn.BCEWithLogitsLoss, **kwargs):
        super(MultiLabelModel, self).__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes

        if model in MultiLabelModel.TORCHVISION_MODEL_NAMES:
            self.model = torch_models.__dict__[model](num_classes = self.num_classes)
        elif model in MultiLabelModel.SEWER_MODEL_NAMES:
            self.model = sewer_models.__dict__[model](num_classes = self.num_classes)
        elif model in MultiLabelModel.MULTILABEL_MODEL_NAMES:
            self.model = ml_models.__dict__[model](num_classes = self.num_classes)
        else:
            raise ValueError("Got model {}, but no such model is in this codebase".format(model))

        self.aux_logits = hasattr(self.model, "aux_logits")

        if self.aux_logits:
            self.train_function = self.aux_loss
        else:
            self.train_function = self.normal_loss
        self.criterion = criterion

        if callable(getattr(self.criterion, "set_device", None)):
            self.criterion.set_device(self.device)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def aux_loss(self, x, y):
        y = y.float()
        y_hat, y_aux_hat = self(x)
        loss = self.criterion(y_hat, y) + 0.4 * self.criterion(y_aux_hat, y)

        return loss
    
    def normal_loss(self, x, y):
        y = y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        return loss


    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.train_function(x, y)
        
        # .log sends to tensorboard/logger, prog_bar also sends to the progress bar
        # result = pl.TrainResult(loss) 
        # result.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=16)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.normal_loss(x, y)

        # lightning monitors 'checkpoint_on' to know when to checkpoint (this is a tensor)
        # module 'pytorch_lightning' has no attribute 'EvalResult'
        # result = pl.EvalResult(checkpoint_on=loss)
        # result.log('val_loss', loss, sync_dist=True)
        self.log('valid_loss', loss, on_step=True, batch_size=16)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.normal_loss(x, y)

        # lightning monitors 'checkpoint_on' to know when to checkpoint (this is a tensor)
        # module 'pytorch_lightning' has no attribute 'EvalResult'
        # result = pl.EvalResult(checkpoint_on=loss)
        # result.log('test_loss', loss, sync_dist=True)
        self.log('test_loss', loss, on_step=True, batch_size=16)


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum, weight_decay = self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30, 60, 80], gamma=0.1)

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument('--model', type=str, default="resnet18", choices=MultiLabelModel.MODEL_NAMES)
        return parser




def main(args):
    pl.seed_everything(1234567890)


    # Init data with transforms
    img_size = 299 if args.model in ["inception_v3", "chen2018_multilabel"] else 224

    train_transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue = 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    eval_transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    if args.training_mode == "e2e":
        dm = MultiLabelDataModule(batch_size = args.batch_size, workers=args.workers, ann_root = args.ann_root, data_root=args.data_root, train_transform=train_transform, eval_transform=eval_transform, only_defects=False)
    elif args.training_mode == "defect":
        dm = MultiLabelDataModule(batch_size = args.batch_size, workers=args.workers, ann_root = args.ann_root, data_root=args.data_root, train_transform=train_transform, eval_transform=eval_transform, only_defects=True)
    elif args.training_mode == "binary":
        dm = BinaryDataModule(batch_size = args.batch_size, workers=args.workers, ann_root = args.ann_root, data_root=args.data_root, train_transform=train_transform, eval_transform=eval_transform)
    elif args.training_mode == "binaryrelevance":
        assert args.br_defect is not None, "Training mode is 'binary_relevance', but no 'br_defect' was stated"
        dm = BinaryRelevanceDataModule(batch_size = args.batch_size, workers=args.workers, ann_root = args.ann_root, data_root=args.data_root, train_transform=train_transform, eval_transform=eval_transform, defect=args.br_defect)
    else:
        raise Exception("Invalid training_mode '{}'".format(args.training_mode))

    dm.prepare_data()
    dm.setup("fit")


    # Init our model
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = dm.class_weights)   

    light_model = MultiLabelModel(num_classes=dm.num_classes, criterion= criterion, **vars(args))

    # train
    prefix = "{}-".format(args.training_mode)
    if args.training_mode == "binaryrelevance":
        prefix += args.br_defect

    logger = TensorBoardLogger(save_dir=args.log_save_dir, name=args.model, version=prefix + "version_" + str(args.log_version))

    logger_path = os.path.join(args.log_save_dir, args.model, prefix + "version_" + str(args.log_version))

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger_path, '{epoch:02d}-{val_loss:.2f}'),
        save_top_k=5,
        save_last = True,
        verbose=False,
        monitor="val_loss",
        mode='min',
        #prefix='',
        #period=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer.from_argparse_args(args, benchmark = True, max_epochs=args.max_epochs, logger=logger, callbacks=[lr_monitor])

    try:
        trainer.fit(light_model, dm)
    except Exception as e:
        print(e)
        with open(os.path.join(logger_path, "error.txt"), "w") as f:
            f.write(str(e))

def run_cli():
    # add PROGRAM level args
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='sewer_env') # default='Pytorch-Lightning'
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default="D:/Research/3-code/sewer-ml/annotations")
    parser.add_argument('--data_root', type=str, default="D:/Research/3-code/sewer-ml/Data/train13")
    parser.add_argument('--batch_size', type=int, default=16, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--log_save_dir', type=str, default="D:/Research/3-code/sewer-ml/logs")
    parser.add_argument('--log_version', type=int, default=1)
    parser.add_argument('--training_mode', type=str, default="e2e", choices=["e2e", "binary", "binaryrelevance", "defect"])
    parser.add_argument('--br_defect', type=str, default=None, choices=[None, "RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"])
    

    # add TRAINER level args
    parser = pl.Trainer.add_argparse_args(parser)

    # add MODEL level args
    parser = MultiLabelModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # Adjust learning rate to amount of nodes/GPUs
    args.workers =  max(0, min(8, 4*args.gpus))
    args.learning_rate = args.learning_rate * (args.gpus * args.num_nodes * args.batch_size) / 256

    main(args)

if __name__ == "__main__":
    run_cli()