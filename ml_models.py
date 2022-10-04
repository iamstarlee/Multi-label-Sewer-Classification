from multilabel_models.tresnet_model import TResNet
from multilabel_models.gcn_models import ResNet_GCNN, ResNet_KSSNET
from multilabel_models.create_adjacency_matrix import normalize_adjacency_matrix
import torchvision.models as torch_models

import torch
import numpy as np


def tresnet_m(num_classes, pretrained = False, **kwargs):
    """ Constructs a medium TResnet model.
    """
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=3,
                    remove_aa_jit=True)
    
    if pretrained:
        # Load state dicts!
        pass

    return model


def tresnet_l(num_classes, pretrained = False, **kwargs):
    """ Constructs a large TResnet model.
    """
    model = TResNet(layers=[4, 5, 18, 3], num_classes=num_classes, in_chans=3, width_factor=1.2,
                    remove_aa_jit=True)
    
    if pretrained:
        # Load state dicts!
        pass

    return model


def tresnet_xl(num_classes, pretrained = False, **kwargs):
    """ Constructs an extra-large TResnet model.
    """
    model = TResNet(layers=[4, 5, 24, 3], num_classes=num_classes, in_chans=3, width_factor=1.3,
                    remove_aa_jit=True)
    
    if pretrained:
        # Load state dicts!
        pass

    return model


def _getResNetBackbone(arch = "resnet101"):

    if arch.lower() == "resnet50":
        backbone = torch_models.resnet50(num_classes=1)
    elif arch.lower() == "resnet101":
        backbone = torch_models.resnet101(num_classes=1)
    elif arch.lower() == "resnet152":
        backbone = torch_models.resnet152(num_classes=1)
    else:
        raise Exception("Invalid architecture passed ({}). Only ResNet50, ResNet101, or ResNet152 is allowed".format(arch))

    return backbone



def resnet_mlgcn(num_classes, pretrained=False, backbone_arch= "resnet101", adj_path ="./adjacency_matrices/adj_reweighted_mlgcn.npy", word_path = "./word_embeddings/one_hot.npy", **kwargs):
    
    backbone = _getResNetBackbone(backbone_arch)

    adj_mat = np.load(adj_path)
    adj_mat = normalize_adjacency_matrix(adj_mat)
    adj_mat = torch.from_numpy(adj_mat)
    
    word_embeddings = np.load(word_path)
    word_embeddings = torch.from_numpy(word_embeddings)

    assert num_classes == adj_mat.shape[0], "Number of classes does not match dimensionality of adjaceny matrix: NUmber of classes: {}  Adjacency Matrix shape: {}".format(num_classes, adj_mat.shape)
    assert num_classes == word_embeddings.shape[0], "Number of classes does not match dimensionality of Word embedding matrix: NUmber of classes: {}  Word embedding Matrix shape: {}".format(num_classes, word_embeddings.shape)
    assert adj_mat.shape[0] == adj_mat.shape[1], "Adjacecny matrix is not square: {}".format(adj_mat.shape)

    adj_mat = adj_mat.float()
    word_embeddings = word_embeddings.float()

    model = ResNet_GCNN(backbone, num_classes, word_embeddings, adj_mat)

    if pretrained:
        # Load state dicts!
        pass
    
    return model

    

def resnet_kssnet(num_classes, pretrained=False, backbone_arch= "resnet101", adj_path ="./adjacency_matrices/adj_ks.npy", word_path = "./word_embeddings/one_hot.npy", **kwargs):
    
    backbone = _getResNetBackbone(backbone_arch)

    adj_mat = np.load(adj_path)
    adj_mat = normalize_adjacency_matrix(adj_mat)
    adj_mat = torch.from_numpy(adj_mat)

    word_embeddings = np.load(word_path)
    word_embeddings = torch.from_numpy(word_embeddings)

    assert num_classes == adj_mat.shape[0], "Number of classes does not match dimensionality of adjacency matrix: NUmber of classes: {}  Adjacency Matrix shape: {}".format(num_classes, adj_mat.shape)
    assert num_classes == word_embeddings.shape[0], "Number of classes does not match dimensionality of Word embedding matrix: NUmber of classes: {}  Word embedding Matrix shape: {}".format(num_classes, word_embeddings.shape)
    assert adj_mat.shape[0] == adj_mat.shape[1], "Adjacency matrix is not square: {}".format(adj_mat.shape)

    adj_mat = adj_mat.float()
    word_embeddings = word_embeddings.float()

    model = ResNet_KSSNET(backbone, num_classes, word_embeddings, adj_mat)

    if pretrained:
        # Load state dicts!
        pass
    
    return model