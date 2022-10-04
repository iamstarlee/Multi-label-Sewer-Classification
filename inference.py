import os
import numpy as np
from argparse import ArgumentParser
from torchvision import models as torch_models
from torchvision import transforms
from collections import OrderedDict
import pandas as pd
import torch

from dataloader import MultiLabelDatasetInference
from torch.utils.data import DataLoader

import torch.nn as nn

import sewer_models
import ml_models


TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__ if name.islower() and not name.startswith("__") and callable(torch_models.__dict__[name]))
SEWER_MODEL_NAMES = sorted(name for name in sewer_models.__dict__ if name.islower() and not name.startswith("__") and callable(sewer_models.__dict__[name]))
MULTILABEL_MODEL_NAMES = sorted(name for name in ml_models.__dict__ if name.islower() and not name.startswith("__") and callable(ml_models.__dict__[name]))
MODEL_NAMES =  TORCHVISION_MODEL_NAMES + SEWER_MODEL_NAMES + MULTILABEL_MODEL_NAMES


def evaluate(dataloader, model, device):
    model.eval()

    sigmoidPredictions = None
    imgPathsList = []

    sigmoid = nn.Sigmoid()

    dataLen = len(dataloader)
    
    with torch.no_grad():
        for i, (images, imgPaths) in enumerate(dataloader):
            if i % 100 == 0:
                print("{} / {}".format(i, dataLen))

            images = images.to(device)

            output = model(images)            

            sigmoidOutput = sigmoid(output).detach().cpu().numpy()

            if sigmoidPredictions is None:
                sigmoidPredictions = sigmoidOutput
            else:
                sigmoidPredictions = np.vstack((sigmoidPredictions, sigmoidOutput))

            imgPathsList.extend(list(imgPaths))
    return sigmoidPredictions, imgPathsList


def load_model(model_path, best_weights=False):

    if best_weights:
        if not os.path.isfile(model_path):
            raise ValueError("The provided path does not lead to a valid file: {}".format(model_path))
        last_ckpt_path = model_path
    else:
        last_ckpt_path = os.path.join(model_path, "last.ckpt")
        if not os.path.isfile(last_ckpt_path):
            raise ValueError("The provided directory path does not contain a 'last.ckpt' file: {}".format(model_path))
    
    model_last_ckpt = torch.load(last_ckpt_path)
    

    model_name = model_last_ckpt["hyper_parameters"]["model"]
    num_classes = model_last_ckpt["hyper_parameters"]["num_classes"]
    training_mode = model_last_ckpt["hyper_parameters"]["training_mode"]
    br_defect = model_last_ckpt["hyper_parameters"]["br_defect"]
    
    # Load best checkpoint
    best_model = model_last_ckpt
    # if best_weights:
    #     best_model = model_last_ckpt
    # else:
    #     best_model_path = model_last_ckpt["checkpoint_callback_best_model_path"]
    #     best_model = torch.load(best_model_path)

    best_model_state_dict = best_model["state_dict"]

    updated_state_dict = OrderedDict()
    for k,v in best_model_state_dict.items():
        name = k.replace("model.", "")
        if "criterion" in name:
            continue

        updated_state_dict[name] = v

    return updated_state_dict, model_name, num_classes, training_mode, br_defect


def run_inference(args):
    
    ann_root = args["ann_root"]
    data_root = args["data_root"]
    model_path = args["model_path"]
    outputPath = args["results_output"]
    best_weights = args["best_weights"]
    # best_weights = False
    split = args["split"]
    
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
  
    updated_state_dict, model_name, num_classes, training_mode, br_defect = load_model(model_path, best_weights)

    if "model_version" not in args.keys():
        model_version = model_name
    else:
        model_version = args["model_version"]

    # Init model
    if model_name in TORCHVISION_MODEL_NAMES:
        model = torch_models.__dict__[model_name](num_classes = num_classes)
    elif model_name in SEWER_MODEL_NAMES:
        model = sewer_models.__dict__[model_name](num_classes = num_classes)
    elif model_name in MULTILABEL_MODEL_NAMES:
        model = ml_models.__dict__[model_name](num_classes = num_classes)
    else:
        raise ValueError("Got model {}, but no such model is in this codebase".format(model_name))

    model.load_state_dict(updated_state_dict)
    
    # initialize dataloaders
    img_size = 299 if model in ["inception_v3", "chen2018_multilabel"] else 224
    
    eval_transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
        ])

        
    dataset = MultiLabelDatasetInference(ann_root, data_root, split=split, transform=eval_transform, onlyDefects=False)
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], num_workers = args["workers"], pin_memory=True)

    if training_mode in ["e2e", "defect"]:
        labelNames = dataset.LabelNames
    elif training_mode == "binary":
        labelNames = ["Defect"]
    elif training_mode == "binaryrelevance":
        labelNames = [br_defect]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Validation results
    print("VALIDATION")
    sigmoid_predictions, val_imgPaths = evaluate(dataloader, model, device)

    sigmoid_dict = {}
    sigmoid_dict["Filename"] = val_imgPaths
    for idx, header in enumerate(labelNames):
        sigmoid_dict[header] = sigmoid_predictions[:,idx]

    sigmoid_df = pd.DataFrame(sigmoid_dict)
    sigmoid_df.to_csv(os.path.join(outputPath, "{}_{}_sigmoid.csv".format(model_version, split.lower())), sep=",", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='pytorch_gpu')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default="D:\\Research\\3-code\\sewer-ml\\annotations")
    parser.add_argument('--data_root', type=str, default="D:\\Research\\3-code\\sewer-ml\\Data\\val13")
    parser.add_argument('--batch_size', type=int, default=16, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument("--model_path", type=str, default="D:\\Research\\3-code\\sewer-ml\\logs\\xie2019_binary\\binary-version_1\\checkpoints")
    parser.add_argument("--best_weights", action="store_true", help="If true 'model_path' leads to a specific weight file. If False it leads to the output folder of lightning_trainer where the last.ckpt file is used to read the best model weights.")
    parser.add_argument("--results_output", type=str, default = "D:\\Research\\3-code\\sewer-ml\\results")
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])



    args = vars(parser.parse_args())

    run_inference(args)